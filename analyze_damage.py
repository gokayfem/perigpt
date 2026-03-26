"""
Analyze whether peridynamic damage correlates with domain boundaries.

This is the KEY experiment for the paper: if damage is high at domain
transitions and low within domains, the theory is validated.

Usage:
    python analyze_damage.py --out_dir=out-mixed-peri-h128
"""

import os
import pickle
import numpy as np
import torch

from model import GPTConfig, GPT, PeriDynamicAttention, StatePeriDynamicAttention

# -----------------------------------------------------------------------------
out_dir = 'out-mixed-peri-h128'
dataset = 'mixed_domain'
n_samples = 50         # number of sequences to analyze
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
# -----------------------------------------------------------------------------

exec(open('configurator.py').read())


def load_model(out_dir):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    for k, v in list(state_dict.items()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, gptconf


def get_per_position_damage(model, x):
    """
    Run forward pass and extract per-position damage from each layer.

    Returns: dict of layer_idx -> (T, delta) damage matrix for each head averaged.
    """
    B, T = x.size()
    config = model.config
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    h = model.transformer.drop(
        model.transformer.wte(x) + model.transformer.wpe(pos))

    layer_damages = {}

    for i, block in enumerate(model.transformer.h):
        # Get the attention module
        attn = block.attn if hasattr(block, 'attn') else None
        norm = block.ln_1 if hasattr(block, 'ln_1') else (
            block.attn_norm if hasattr(block, 'attn_norm') else None)

        if attn is None or norm is None:
            h = block(h) if not hasattr(block, 'forward') else h
            continue

        is_peri = isinstance(attn, (PeriDynamicAttention, StatePeriDynamicAttention))
        if not is_peri:
            if hasattr(attn, 'peri_attn'):
                attn = attn.peri_attn
                is_peri = True

        if is_peri:
            with torch.no_grad():
                normed = norm(h)
                bd = attn.bond_dim
                delta = min(attn.horizon, T)
                nh = attn.n_head

                disp = attn.W_disp(normed).view(B, T, nh, bd).permute(0, 2, 1, 3)
                disp_win = attn._build_windows(disp, delta)
                strain = disp_win - disp.unsqueeze(3)

                if hasattr(attn, 'strain_fused'):
                    fused = attn.strain_fused(strain)
                    _, damage_feats = fused.chunk(2, dim=-1)
                    damage = torch.sigmoid(attn.damage_out(torch.nn.functional.gelu(damage_feats)).squeeze(-1))
                else:
                    damage_act = torch.nn.functional.gelu(attn.damage_proj(strain))
                    damage = torch.sigmoid(attn.damage_out(damage_act).squeeze(-1))
                # damage: (B, nh, T, delta)

                # Average over batch and heads
                cmask = attn._causal_mask(T, delta, device).float()
                damage_masked = damage * cmask.unsqueeze(0).unsqueeze(0)
                # Mean damage per position (average over delta and heads)
                per_pos = damage_masked.mean(dim=(0, 1, 3))  # (T,)
                layer_damages[i] = per_pos.cpu().numpy()

        # Advance through block
        h = block(h)

    return layer_damages


def detect_domain_boundaries(labels, block_size):
    """
    Find positions where domain changes within each block_size window.
    Returns boolean array: True at domain boundary positions.
    """
    boundaries = np.zeros(len(labels), dtype=bool)
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries[i] = True
    return boundaries


def main():
    print(f"Loading model from {out_dir}...")
    model, config = load_model(out_dir)
    block_size = config.block_size

    print(f"Loading domain labels...")
    data_dir = os.path.join('data', dataset)
    with open(os.path.join(data_dir, 'domain_labels.pkl'), 'rb') as f:
        labels_meta = pickle.load(f)

    val_labels = labels_meta['val_labels']

    # Load val data
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # Sample random windows
    print(f"Analyzing {n_samples} sequences...")
    all_boundary_damages = []
    all_interior_damages = []

    for s in range(n_samples):
        start = np.random.randint(0, len(val_data) - block_size)
        x = torch.from_numpy(val_data[start:start + block_size].astype(np.int64)).unsqueeze(0).to(device)

        # Get damage per position
        layer_damages = get_per_position_damage(model, x)

        if not layer_damages:
            print("  No peridynamic layers found!")
            return

        # Average damage across all layers
        avg_damage = np.mean(list(layer_damages.values()), axis=0)  # (T,)

        # Get domain labels for this window
        window_labels = val_labels[start:start + block_size]

        # Find boundary positions (where domain changes)
        boundaries = detect_domain_boundaries(window_labels, block_size)

        # Separate boundary vs interior damage
        # Use a window around each boundary (±3 positions)
        boundary_zone = np.zeros(block_size, dtype=bool)
        for i in range(block_size):
            if boundaries[i]:
                for j in range(max(0, i - 3), min(block_size, i + 4)):
                    boundary_zone[j] = True

        interior_zone = ~boundary_zone

        if boundary_zone.sum() > 0:
            all_boundary_damages.append(avg_damage[boundary_zone].mean())
        if interior_zone.sum() > 0:
            all_interior_damages.append(avg_damage[interior_zone].mean())

    # Results
    boundary_dmg = np.mean(all_boundary_damages)
    interior_dmg = np.mean(all_interior_damages)
    boundary_std = np.std(all_boundary_damages)
    interior_std = np.std(all_interior_damages)

    print(f"\n{'=' * 60}")
    print(f"DAMAGE ANALYSIS RESULTS")
    print(f"{'=' * 60}")
    print(f"  Model: {out_dir}")
    print(f"  Sequences analyzed: {n_samples}")
    print(f"  Layers with damage: {list(layer_damages.keys())}")
    print(f"")
    print(f"  Mean damage at domain BOUNDARIES:  {boundary_dmg:.4f} (±{boundary_std:.4f})")
    print(f"  Mean damage at domain INTERIOR:    {interior_dmg:.4f} (±{interior_std:.4f})")
    print(f"  Ratio (boundary / interior):       {boundary_dmg / (interior_dmg + 1e-8):.2f}x")
    print(f"")
    if boundary_dmg > interior_dmg:
        print(f"  >> DAMAGE IS HIGHER AT BOUNDARIES — theory supported")
    else:
        print(f"  >> Damage is NOT higher at boundaries — theory not supported")

    # Per-layer breakdown
    print(f"\n  Per-layer mean damage:")
    for layer_idx, dmg in sorted(layer_damages.items()):
        print(f"    Layer {layer_idx}: {dmg.mean():.4f}")

    # Save results for plotting
    results = {
        'boundary_damages': all_boundary_damages,
        'interior_damages': all_interior_damages,
        'boundary_mean': boundary_dmg,
        'interior_mean': interior_dmg,
        'layer_damages': layer_damages,
        'n_samples': n_samples,
        'out_dir': out_dir,
    }
    results_path = os.path.join(out_dir, 'damage_analysis.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Saved detailed results to {results_path}")


if __name__ == '__main__':
    main()
