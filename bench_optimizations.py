"""
Benchmark 20 optimization variants for peridynamic attention.
Run on MPS: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python bench_optimizations.py
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
B, T, C, nh, hs, bd, delta = 4, 256, 384, 6, 64, 32, 128
dtype = torch.float32

results = []


def bench(label, fn, n_warmup=3, n_runs=10):
    for _ in range(n_warmup):
        fn()
    if device == 'mps':
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if device == 'mps':
        torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / n_runs * 1000
    results.append((label, ms))
    print(f'  {label:55s} {ms:8.2f} ms')
    return ms


# Shared data
torch.manual_seed(42)
disp = torch.randn(B, nh, T, bd, device=device)
val = torch.randn(B, nh, T, hs, device=device)
disp_padded = F.pad(disp, (0, 0, delta - 1, 0)).contiguous()
val_padded = F.pad(val, (0, 0, delta - 1, 0)).contiguous()

# Shared weights
W_fused = nn.Linear(bd, bd * 2, bias=True).to(device)
W_pos = nn.Linear(bd, bd, bias=False).to(device)
W_bond = nn.Linear(bd, 1, bias=True).to(device)
W_dmg = nn.Linear(bd, 1, bias=True).to(device)
rel_emb = nn.Embedding(delta, bd).to(device)
pos_feat = W_pos(rel_emb(torch.arange(delta, device=device)))

# Causal mask
t_idx = torch.arange(T, device=device).unsqueeze(1)
w_idx = torch.arange(delta, device=device).unsqueeze(0)
cmask = (t_idx - delta + 1 + w_idx) >= 0

print(f'Device: {device}, B={B}, T={T}, nh={nh}, hs={hs}, bd={bd}, delta={delta}\n')
print('=' * 75)
print('STRAIN COMPUTATION (48% of total — the #1 target)')
print('=' * 75)

# Baseline: unfold + subtract
def strain_baseline():
    padded = F.pad(disp, (0, 0, delta - 1, 0))
    win = padded.unfold(2, delta, 1).permute(0, 1, 2, 4, 3)
    return win - disp.unsqueeze(3)
bench('[1] BASELINE: unfold + permute + subtract', strain_baseline)

# Opt 1: as_strided window (avoid permute copy)
def strain_strided():
    BN = B * nh
    dp = disp_padded.reshape(BN, T + delta - 1, bd).contiguous()
    s = dp.stride()
    win = dp.as_strided((BN, T, delta, bd), (s[0], s[1], s[1], s[2]))
    di = disp.reshape(BN, T, bd).unsqueeze(2)
    return win - di
bench('[2] as_strided window + subtract', strain_strided)

# Opt 2: compute strain without explicit window — via index_select
def strain_index_select():
    BN = B * nh
    dp = disp_padded.reshape(BN, T + delta - 1, bd)
    # Build index: for each (t, j), the source position is t + j
    idx = torch.arange(T, device=device).unsqueeze(1) + torch.arange(delta, device=device).unsqueeze(0)
    idx_flat = idx.reshape(-1)
    gathered = dp[:, idx_flat].reshape(BN, T, delta, bd)
    di = disp.reshape(BN, T, 1, bd)
    return gathered - di
bench('[3] index gather + subtract', strain_index_select)

# Opt 3: fuse strain computation into the projection — skip explicit strain tensor
# Instead of: strain = win - center; fused(strain) = strain @ W
# Do: fused = win @ W - center @ W (linearity!)
def strain_fused_skip():
    BN = B * nh
    dp = disp_padded.reshape(BN, T + delta - 1, bd).contiguous()
    s = dp.stride()
    win = dp.as_strided((BN, T, delta, bd), (s[0], s[1], s[1], s[2]))
    # Project win and center separately, then subtract
    # win @ W_fused^T: (BN, T, delta, bd) @ (bd, 2*bd) -> (BN, T, delta, 2*bd)
    win_proj = F.linear(win, W_fused.weight, None)
    center_proj = F.linear(disp.reshape(BN, T, bd), W_fused.weight, None)
    # strain_proj = win_proj - center_proj + bias
    result = win_proj - center_proj.unsqueeze(2) + W_fused.bias
    return result
bench('[4] SKIP STRAIN: project(win) - project(center) (linearity)', strain_fused_skip)

# Opt 4: float16 strain
def strain_fp16():
    d16 = disp.half()
    padded = F.pad(d16, (0, 0, delta - 1, 0))
    win = padded.unfold(2, delta, 1).permute(0, 1, 2, 4, 3)
    return win - d16.unsqueeze(3)
bench('[5] float16 strain computation', strain_fp16)

# Opt 5: Contiguous before subtract
def strain_contig():
    padded = F.pad(disp, (0, 0, delta - 1, 0))
    win = padded.unfold(2, delta, 1).permute(0, 1, 2, 4, 3).contiguous()
    return win - disp.unsqueeze(3)
bench('[6] unfold + contiguous() before subtract', strain_contig)

print()
print('=' * 75)
print('CAUSAL MASK (9% of total)')
print('=' * 75)

# Baseline
def mask_baseline():
    ti = torch.arange(T, device=device).unsqueeze(1)
    wi = torch.arange(delta, device=device).unsqueeze(0)
    return (ti - delta + 1 + wi) >= 0
bench('[7] BASELINE: arange + broadcast compare', mask_baseline)

# Opt: cache the mask
_cached_mask = cmask
def mask_cached():
    return _cached_mask
bench('[8] cached mask (precomputed)', mask_cached)

# Opt: tril-based mask
def mask_tril():
    return torch.ones(T, delta, device=device, dtype=torch.bool).tril(delta - 1)
bench('[9] tril-based mask', mask_tril)

print()
print('=' * 75)
print('GELU + POSITION ADDITION (8% of total)')
print('=' * 75)

strain = strain_baseline()
fused_out = W_fused(strain)
bf, df = fused_out.chunk(2, dim=-1)

# Baseline
def gelu_baseline():
    return F.gelu(bf + pos_feat)
bench('[10] BASELINE: gelu(bond_feats + pos_feat)', gelu_baseline)

# Opt: tanh gelu approx (already default in PyTorch)
def gelu_approx():
    return F.gelu(bf + pos_feat, approximate='tanh')
bench('[11] gelu approximate=tanh', gelu_approx)

# Opt: relu instead of gelu
def relu_variant():
    return F.relu(bf + pos_feat)
bench('[12] relu instead of gelu (different activation)', relu_variant)

# Opt: pre-add pos_feat to bond_feats via broadcasting before gelu
# (same as baseline but pre-expand)
def gelu_precompute():
    combined = bf + pos_feat  # ensure broadcast happens once
    return F.gelu(combined)
bench('[13] pre-add then gelu (same but explicit)', gelu_precompute)

print()
print('=' * 75)
print('BOND + DAMAGE OUTPUT (combined 5%)')
print('=' * 75)

bond_act = F.gelu(bf + pos_feat)
dmg_act = F.gelu(df)

# Baseline: two separate linears
def bd_out_baseline():
    bl = W_bond(bond_act).squeeze(-1)
    dm = torch.sigmoid(W_dmg(dmg_act).squeeze(-1))
    return bl - dm * 10.0
bench('[14] BASELINE: separate bond_out + damage_out', bd_out_baseline)

# Opt: fuse into one (bd,2) linear
W_bd_fused = nn.Linear(bd, 2, bias=True).to(device)
def bd_out_fused():
    # Stack activations for one matmul
    # but they have different inputs (bond_act vs dmg_act)...
    # Can't directly fuse unless we cat first
    cat_act = torch.cat([bond_act, dmg_act], dim=-1)  # (B,nh,T,d,2*bd)
    # Need a (2*bd, 2) weight... this is a different opt
    return cat_act  # placeholder
bench('[15] cat(bond,dmg) for potential fused output', lambda: torch.cat([bond_act, dmg_act], dim=-1))

# Opt: use einsum for dot product instead of Linear
w_bond_vec = W_bond.weight.squeeze()  # (bd,)
w_dmg_vec = W_dmg.weight.squeeze()
def bd_out_einsum():
    bl = torch.einsum('...d,d->...', bond_act, w_bond_vec) + W_bond.bias.squeeze()
    dm = torch.sigmoid(torch.einsum('...d,d->...', dmg_act, w_dmg_vec) + W_dmg.bias.squeeze())
    return bl - dm * 10.0
bench('[16] einsum dot product instead of nn.Linear', bd_out_einsum)

print()
print('=' * 75)
print('VALUE AGGREGATION (10% of total)')
print('=' * 75)

weights = F.softmax(torch.randn(B, nh, T, delta, device=device), dim=-1)

# Baseline: strided matmul
def val_agg_strided():
    BN = B * nh
    vf = val_padded.reshape(BN, T + delta - 1, hs).contiguous()
    s = vf.stride()
    vw = vf.as_strided((BN, T, delta, hs), (s[0], s[1], s[1], s[2]))
    wf = weights.reshape(BN, T, 1, delta)
    return torch.matmul(wf, vw).squeeze(2)
bench('[17] BASELINE: strided matmul', val_agg_strided)

# Opt: shift-accumulate loop (was faster on CPU, check MPS)
def val_agg_loop():
    output = torch.zeros(B, nh, T, hs, device=device, dtype=dtype)
    for j in range(delta):
        w_j = weights[:, :, :, j].unsqueeze(-1)
        output = output + w_j * val_padded[:, :, j:j + T]
    return output
bench('[18] shift-accumulate loop', val_agg_loop)

# Opt: unfold + einsum (original approach)
def val_agg_unfold_einsum():
    win = val_padded.unfold(2, delta, 1).permute(0, 1, 2, 4, 3)
    return torch.einsum('bntd,bntde->bnte', weights, win)
bench('[19] unfold + einsum (original)', val_agg_unfold_einsum)

# Opt: bmm with explicit reshape
def val_agg_bmm():
    BN = B * nh
    vf = val_padded.reshape(BN, T + delta - 1, hs).contiguous()
    s = vf.stride()
    vw = vf.as_strided((BN, T, delta, hs), (s[0], s[1], s[1], s[2]))
    vw_c = vw.reshape(BN * T, delta, hs)
    wf = weights.reshape(BN * T, 1, delta)
    return torch.bmm(wf, vw_c).reshape(BN, T, hs)
bench('[20] bmm with BN*T batch', val_agg_bmm)

print()
print('=' * 75)
print('FULL PIPELINE OPTIMIZATIONS')
print('=' * 75)

# Opt: skip strain entirely via linearity
def full_skip_strain():
    BN = B * nh
    dp = disp_padded.reshape(BN, T + delta - 1, bd).contiguous()
    s = dp.stride()
    win = dp.as_strided((BN, T, delta, bd), (s[0], s[1], s[1], s[2]))
    center = disp.reshape(BN, T, bd)
    # Project window and center separately: strain_fused(strain) = W @ (win - center)
    # = W @ win - W @ center (linearity of linear transform)
    win_proj = F.linear(win, W_fused.weight, None)         # (BN, T, delta, 2*bd)
    center_proj = F.linear(center, W_fused.weight, None)   # (BN, T, 2*bd)
    fused_result = win_proj - center_proj.unsqueeze(2) + W_fused.bias  # broadcast
    bf2, df2 = fused_result.chunk(2, dim=-1)
    ba2 = F.gelu(bf2 + pos_feat)
    da2 = F.gelu(df2)
    bl2 = torch.einsum('...d,d->...', ba2, w_bond_vec) + W_bond.bias.squeeze()
    dl2 = torch.sigmoid(torch.einsum('...d,d->...', da2, w_dmg_vec) + W_dmg.bias.squeeze())
    logits2 = bl2 - dl2 * 10.0
    return logits2.reshape(B, nh, T, delta)
bench('[21] FULL: skip strain via linearity (project win/center separately)', full_skip_strain)

# Baseline full pipeline for reference
def full_baseline():
    padded = F.pad(disp, (0, 0, delta - 1, 0))
    win = padded.unfold(2, delta, 1).permute(0, 1, 2, 4, 3)
    s = win - disp.unsqueeze(3)
    f = W_fused(s)
    bf3, df3 = f.chunk(2, dim=-1)
    ba3 = F.gelu(bf3 + pos_feat)
    da3 = F.gelu(df3)
    bl3 = W_bond(ba3).squeeze(-1)
    dl3 = torch.sigmoid(W_dmg(da3).squeeze(-1))
    return bl3 - dl3 * 10.0
bench('[22] FULL: baseline pipeline', full_baseline)

print()
print('=' * 75)
print('SUMMARY (sorted by speed)')
print('=' * 75)
results.sort(key=lambda x: x[1])
for label, ms in results:
    print(f'  {ms:8.2f} ms  {label}')
