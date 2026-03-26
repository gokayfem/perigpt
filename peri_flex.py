"""
FlexAttention-based peridynamic attention.

Uses PyTorch's flex_attention (torch.nn.attention.flex_attention) which
provides flash-attention-level speed for custom attention patterns.

Strategy:
  1. Pre-compute strain-based bond scores + damage via cuBLAS (the 5D matmuls)
  2. Hand the scores to flex_attention with a sliding window block_mask
  3. flex_attention handles softmax + value aggregation at flash speed

This avoids building val_win entirely and uses the optimized flash
attention kernel for the heavy part (memory-bound softmax @ V).
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
    )
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False


class PeriDynamicAttentionFlex(nn.Module):
    """
    Peridynamic attention using FlexAttention for the softmax + value step.

    The score computation (strain → projection → bond/damage) uses standard
    PyTorch ops backed by cuBLAS. The softmax + value aggregation uses
    flex_attention's flash-attention-optimized kernel.

    Falls back to standard PeriDynamicAttention on older PyTorch.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.horizon = getattr(config, 'horizon', 32)
        self.dropout = config.dropout

        hs = self.head_size
        bond_dim_ratio = getattr(config, 'bond_dim_ratio', 4)
        self.bond_dim = max(hs // bond_dim_ratio, 16)
        bd = self.bond_dim

        # Displacement and value projections
        self.W_disp = nn.Linear(config.n_embd, config.n_head * bd, bias=config.bias)
        self.W_val = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Relative position embedding
        self.rel_pos_emb = nn.Embedding(self.horizon, bd)

        # Fused bond + damage projection
        self.strain_fused = nn.Linear(bd, bd * 2, bias=True)
        self.pos_proj = nn.Linear(bd, bd, bias=False)
        self.bond_out = nn.Linear(bd, 1, bias=config.bias)
        self.damage_out = nn.Linear(bd, 1, bias=True)
        nn.init.constant_(self.damage_out.bias, -3.0)

        # Output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Block mask (created once, cached)
        self._block_mask_cache = {}

    def _get_block_mask(self, B, T, delta, device):
        """Create sliding window block mask for flex_attention."""
        key = (B, self.n_head, T, delta)
        if key not in self._block_mask_cache:
            def sliding_window_causal(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                in_window = (q_idx - kv_idx) < delta
                return causal & in_window

            mask = create_block_mask(
                sliding_window_causal,
                B=B, H=self.n_head, Q_LEN=T, KV_LEN=T,
                device=device,
            )
            self._block_mask_cache[key] = mask
        return self._block_mask_cache[key]

    def _compute_peri_scores(self, disp, delta, device):
        """
        Compute peridynamic bond scores as a (B, nh, T, T) sparse score tensor.

        Returns a function suitable for flex_attention's score_mod that maps
        (q_idx, kv_idx) to the pre-computed peridynamic score.
        """
        B, nh, T, bd = disp.shape

        # Build displacement windows and compute strain
        padded = F.pad(disp, (0, 0, delta - 1, 0))
        win = padded.unfold(2, delta, 1).permute(0, 1, 2, 4, 3)  # (B, nh, T, delta, bd)
        strain = win - disp.unsqueeze(3)

        # Position features
        rel_ids = torch.arange(delta, device=device)
        pos_feat = self.pos_proj(self.rel_pos_emb(rel_ids))

        # Fused projection
        fused = self.strain_fused(strain)
        bond_feats, damage_feats = fused.chunk(2, dim=-1)

        bond_logits = self.bond_out(F.gelu(bond_feats + pos_feat)).squeeze(-1)  # (B, nh, T, delta)
        damage = torch.sigmoid(self.damage_out(F.gelu(damage_feats)).squeeze(-1))

        # Final scores: (B, nh, T, delta)
        peri_scores = bond_logits - damage * 10.0

        # Scatter into full (B, nh, T, T) attention score matrix
        # For each query position t, scores go to kv positions [t-delta+1, ..., t]
        scores_full = torch.full((B, nh, T, T), float('-inf'), device=device, dtype=disp.dtype)
        for j in range(delta):
            kv_pos = torch.arange(T, device=device) - delta + 1 + j  # source positions
            valid = kv_pos >= 0
            q_pos = torch.arange(T, device=device)
            scores_full[:, :, q_pos[valid], kv_pos[valid]] = peri_scores[:, :, q_pos[valid], j]

        return scores_full

    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head
        hs = self.head_size
        bd = self.bond_dim
        delta = min(self.horizon, T)

        disp = self.W_disp(x).view(B, T, nh, bd).permute(0, 2, 1, 3)
        val = self.W_val(x).view(B, T, nh, hs).permute(0, 2, 1, 3)

        if x.is_cuda and HAS_FLEX:
            # Compute peridynamic scores
            peri_scores = self._compute_peri_scores(disp, delta, x.device)

            # Create score_mod that returns our pre-computed scores
            # flex_attention calls: final_score = score_mod(q @ k^T / sqrt(d), b, h, q_idx, kv_idx)
            # We ignore the input score and return our peridynamic score
            def score_mod(score, b, h, q_idx, kv_idx):
                return peri_scores[b, h, q_idx, kv_idx]

            block_mask = self._get_block_mask(B, T, delta, x.device)

            # Use dummy Q, K (scores come entirely from score_mod)
            # Q, K shape must be (B, nh, T, head_dim) for flex_attention
            q_dummy = torch.zeros(B, nh, T, hs, device=x.device, dtype=x.dtype)
            k_dummy = torch.zeros(B, nh, T, hs, device=x.device, dtype=x.dtype)

            output = flex_attention(
                q_dummy, k_dummy, val,
                block_mask=block_mask,
                score_mod=score_mod,
            )
        else:
            # Fallback: standard PyTorch path
            output = self._pytorch_fallback(disp, val, delta, x)

        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        output = self.resid_dropout(self.c_proj(output))
        return output

    def _pytorch_fallback(self, disp, val, delta, x):
        """Same as PeriDynamicAttention forward."""
        B, nh, T, bd = disp.shape
        hs = val.shape[-1]

        padded = F.pad(disp, (0, 0, delta - 1, 0))
        win = padded.unfold(2, delta, 1).permute(0, 1, 2, 4, 3)
        strain = win - disp.unsqueeze(3)

        rel_ids = torch.arange(delta, device=x.device)
        pos_feat = self.pos_proj(self.rel_pos_emb(rel_ids))
        fused = self.strain_fused(strain)
        bf, df = fused.chunk(2, dim=-1)
        bond_logits = self.bond_out(F.gelu(bf + pos_feat)).squeeze(-1)
        damage = torch.sigmoid(self.damage_out(F.gelu(df)).squeeze(-1))
        bond_logits = bond_logits - damage * 10.0

        t_idx = torch.arange(T, device=x.device).unsqueeze(1)
        w_idx = torch.arange(delta, device=x.device).unsqueeze(0)
        cmask = (t_idx - delta + 1 + w_idx) >= 0
        bond_logits = bond_logits.masked_fill(~cmask.unsqueeze(0).unsqueeze(0), float('-inf'))
        weights = F.softmax(bond_logits, dim=-1)

        val_padded = F.pad(val, (0, 0, delta - 1, 0))
        if x.is_cuda or x.device.type == 'mps':
            BN = B * nh
            vf = val_padded.reshape(BN, T + delta - 1, hs).contiguous()
            s = vf.stride()
            vw = vf.as_strided((BN, T, delta, hs), (s[0], s[1], s[1], s[2]))
            wf = weights.reshape(BN, T, 1, delta)
            output = torch.matmul(wf, vw).squeeze(2).reshape(B, nh, T, hs)
        else:
            output = torch.zeros(B, nh, T, hs, device=x.device, dtype=val.dtype)
            for j in range(delta):
                w_j = weights[:, :, :, j].unsqueeze(-1)
                output = output + w_j * val_padded[:, :, j:j + T]
        return output

    def get_damage_stats(self, x):
        B, T, C = x.size()
        bd = self.bond_dim
        delta = min(self.horizon, T)
        nh = self.n_head
        disp = self.W_disp(x).view(B, T, nh, bd).permute(0, 2, 1, 3)
        padded = F.pad(disp, (0, 0, delta - 1, 0))
        win = padded.unfold(2, delta, 1).permute(0, 1, 2, 4, 3)
        strain = win - disp.unsqueeze(3)
        fused = self.strain_fused(strain)
        _, df = fused.chunk(2, dim=-1)
        damage = torch.sigmoid(self.damage_out(F.gelu(df)).squeeze(-1))
        t_idx = torch.arange(T, device=x.device).unsqueeze(1)
        w_idx = torch.arange(delta, device=x.device).unsqueeze(0)
        cmask = ((t_idx - delta + 1 + w_idx) >= 0).float()
        valid_count = cmask.sum()
        mean_damage = (damage * cmask.unsqueeze(0).unsqueeze(0)).sum() / (
            valid_count * B * nh + 1e-8)
        return mean_damage.item()
