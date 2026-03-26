"""
Triton kernel for peridynamic attention.

Fuses the entire inner loop into a single GPU kernel:
  strain computation -> fused projection -> bond+damage scoring ->
  online softmax -> value accumulation

No intermediate 5D tensors. Peak memory: O(B * nh * T * hs).
Falls back to optimized PyTorch if Triton is unavailable.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════════════════
# Triton Kernel (only compiled on CUDA with Triton installed)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _gelu_approx(x):
        return 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

    @triton.jit
    def _peri_attn_kernel(
        # Data pointers
        disp_ptr, val_ptr, out_ptr,
        # Weight pointers (pre-transposed for row-major access)
        W_fused_ptr, W_fused_bias_ptr,
        W_pos_ptr, rel_pos_ptr,
        W_bond_ptr, W_bond_bias_ptr,
        W_dmg_ptr, W_dmg_bias_ptr,
        # Dimensions
        T: tl.constexpr, HS: tl.constexpr, BD: tl.constexpr,
        DELTA: tl.constexpr, NH: tl.constexpr,
        # Strides for disp (B*nh, T, bd) — pre-reshaped
        stride_dt, stride_dd,
        # Strides for val (B*nh, T, hs) — pre-reshaped
        stride_vt, stride_vd,
        # Strides for out (B*nh, T, hs) — pre-reshaped
        stride_ot, stride_od,
        # Block size for HS dimension
        BLOCK_BD: tl.constexpr,
        BLOCK_HS: tl.constexpr,
    ):
        # One program = one (batch*head, position) pair
        pid_bh = tl.program_id(0)  # batch * head index
        pid_t = tl.program_id(1)   # position index

        bd_offs = tl.arange(0, BLOCK_BD)  # [0, 1, ..., BD-1]
        hs_offs = tl.arange(0, BLOCK_HS)  # [0, 1, ..., HS-1]
        bd_mask = bd_offs < BD
        hs_mask = hs_offs < HS

        # Load center displacement: disp[pid_bh, pid_t, :]
        disp_base = pid_bh * T * BD + pid_t * stride_dt
        center = tl.load(disp_ptr + disp_base + bd_offs * stride_dd, mask=bd_mask, other=0.0)

        # Online softmax state
        m_prev = float('-inf')  # running max
        d_prev = 0.0            # running sum of exp
        acc = tl.zeros((BLOCK_HS,), dtype=tl.float32)  # weighted value accumulator

        for j in range(DELTA):
            nb_t = pid_t - DELTA + 1 + j
            if nb_t >= 0:
                # Load neighbor displacement
                nb_disp_base = pid_bh * T * BD + nb_t * stride_dt
                neighbor = tl.load(disp_ptr + nb_disp_base + bd_offs * stride_dd,
                                   mask=bd_mask, other=0.0)
                strain = neighbor - center  # (BD,)

                # Fused projection: strain @ W_fused -> (2*BD,)
                # W_fused is stored as (BD, 2*BD) after transpose
                bond_feat = tl.zeros((BLOCK_BD,), dtype=tl.float32)
                dmg_feat = tl.zeros((BLOCK_BD,), dtype=tl.float32)
                for k in range(BD):
                    s_k = tl.load(disp_ptr + nb_disp_base + k * stride_dd) - tl.load(disp_ptr + disp_base + k * stride_dd)
                    # Bond part: column k of W_fused rows [0:BD]
                    w_b = tl.load(W_fused_ptr + k * BD * 2 + bd_offs, mask=bd_mask, other=0.0)
                    bond_feat += s_k * w_b
                    # Damage part: column k of W_fused rows [BD:2BD]
                    w_d = tl.load(W_fused_ptr + k * BD * 2 + BD + bd_offs, mask=bd_mask, other=0.0)
                    dmg_feat += s_k * w_d

                # Add fused bias
                bond_feat += tl.load(W_fused_bias_ptr + bd_offs, mask=bd_mask, other=0.0)
                dmg_feat += tl.load(W_fused_bias_ptr + BD + bd_offs, mask=bd_mask, other=0.0)

                # Position projection: rel_pos[j] @ W_pos
                pos_feat = tl.zeros((BLOCK_BD,), dtype=tl.float32)
                for k in range(BD):
                    rp_k = tl.load(rel_pos_ptr + j * BD + k)
                    w_p = tl.load(W_pos_ptr + k * BD + bd_offs, mask=bd_mask, other=0.0)
                    pos_feat += rp_k * w_p

                # Bond score = W_bond @ gelu(bond_feat + pos_feat)
                bond_act = _gelu_approx(bond_feat + pos_feat)
                w_bond = tl.load(W_bond_ptr + bd_offs, mask=bd_mask, other=0.0)
                bond_logit = tl.sum(w_bond * bond_act)
                bond_bias = tl.load(W_bond_bias_ptr)
                bond_logit += bond_bias

                # Damage = sigmoid(W_dmg @ gelu(dmg_feat) + bias)
                dmg_act = _gelu_approx(dmg_feat)
                w_dmg = tl.load(W_dmg_ptr + bd_offs, mask=bd_mask, other=0.0)
                dmg_logit = tl.sum(w_dmg * dmg_act) + tl.load(W_dmg_bias_ptr)
                damage = tl.sigmoid(dmg_logit)

                logit = bond_logit - damage * 10.0

                # Online softmax update (flash attention style)
                m_new = tl.maximum(m_prev, logit)
                scale = tl.math.exp(m_prev - m_new)
                p_new = tl.math.exp(logit - m_new)

                d_prev = d_prev * scale + p_new
                acc = acc * scale

                # Load neighbor value and accumulate
                nb_val_base = pid_bh * T * HS + nb_t * stride_vt
                nb_val = tl.load(val_ptr + nb_val_base + hs_offs * stride_vd,
                                 mask=hs_mask, other=0.0)
                acc += p_new * nb_val

                m_prev = m_new

        # Normalize
        out = acc / (d_prev + 1e-8)

        # Store
        out_base = pid_bh * T * HS + pid_t * stride_ot
        tl.store(out_ptr + out_base + hs_offs * stride_od, out, mask=hs_mask)


def peri_attn_triton_forward(disp, val, attn_module, delta):
    """
    Launch the Triton kernel for peridynamic attention.

    Args:
        disp: (B, nh, T, bd) displacement features
        val: (B, nh, T, hs) value features
        attn_module: the PeriDynamicAttentionTriton module (for accessing weights)
        delta: horizon size

    Returns: (B, nh, T, hs) output
    """
    B, nh, T, bd = disp.shape
    hs = val.shape[-1]

    # Reshape to (B*nh, T, dim) for simpler kernel indexing
    disp_flat = disp.reshape(B * nh, T, bd).contiguous().float()
    val_flat = val.reshape(B * nh, T, hs).contiguous().float()
    out_flat = torch.empty(B * nh, T, hs, device=disp.device, dtype=torch.float32)

    # Pre-compute position embeddings
    rel_ids = torch.arange(delta, device=disp.device)
    rel_pos = attn_module.rel_pos_emb(rel_ids).contiguous().float()

    # Weight access
    W_fused = attn_module.strain_fused.weight.contiguous().float()  # (2*bd, bd)
    W_fused_bias = attn_module.strain_fused.bias.contiguous().float()
    W_pos = attn_module.pos_proj.weight.contiguous().float()  # (bd, bd)
    W_bond = attn_module.bond_out.weight.squeeze(0).contiguous().float()  # (bd,)
    W_bond_bias = attn_module.bond_out.bias.contiguous().float() if attn_module.bond_out.bias is not None else torch.zeros(1, device=disp.device)
    W_dmg = attn_module.damage_out.weight.squeeze(0).contiguous().float()
    W_dmg_bias = attn_module.damage_out.bias.contiguous().float()

    BLOCK_BD = triton.next_power_of_2(bd)
    BLOCK_HS = triton.next_power_of_2(hs)

    grid = (B * nh, T)

    _peri_attn_kernel[grid](
        disp_flat, val_flat, out_flat,
        W_fused, W_fused_bias,
        W_pos, rel_pos,
        W_bond, W_bond_bias,
        W_dmg, W_dmg_bias,
        T, hs, bd, delta, nh,
        disp_flat.stride(1), disp_flat.stride(2),
        val_flat.stride(1), val_flat.stride(2),
        out_flat.stride(1), out_flat.stride(2),
        BLOCK_BD=BLOCK_BD,
        BLOCK_HS=BLOCK_HS,
    )

    return out_flat.reshape(B, nh, T, hs).to(disp.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Module (always importable, falls back to PyTorch on CPU/no-Triton)
# ═══════════════════════════════════════════════════════════════════════════

class PeriDynamicAttentionTriton(nn.Module):
    """
    Drop-in replacement for PeriDynamicAttention using Triton kernel.

    Same weights, same init, same API. Uses fused Triton kernel on CUDA,
    falls back to optimized PyTorch on CPU/MPS.
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

        self.W_disp = nn.Linear(config.n_embd, config.n_head * bd, bias=config.bias)
        self.W_val = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.rel_pos_emb = nn.Embedding(self.horizon, bd)
        self.strain_fused = nn.Linear(bd, bd * 2, bias=True)
        self.pos_proj = nn.Linear(bd, bd, bias=False)
        self.bond_out = nn.Linear(bd, 1, bias=config.bias)
        self.damage_out = nn.Linear(bd, 1, bias=True)
        nn.init.constant_(self.damage_out.bias, -3.0)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _build_windows(self, tensor, delta):
        padded = F.pad(tensor, (0, 0, delta - 1, 0))
        win = padded.unfold(2, delta, 1)
        return win.permute(0, 1, 2, 4, 3)

    def _causal_mask(self, T, delta, device):
        t_idx = torch.arange(T, device=device).unsqueeze(1)
        w_idx = torch.arange(delta, device=device).unsqueeze(0)
        return (t_idx - delta + 1 + w_idx) >= 0

    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head
        hs = self.head_size
        bd = self.bond_dim
        delta = min(self.horizon, T)

        disp = self.W_disp(x).view(B, T, nh, bd).permute(0, 2, 1, 3)
        val = self.W_val(x).view(B, T, nh, hs).permute(0, 2, 1, 3)

        if x.is_cuda and HAS_TRITON and not self.training:
            # Triton path — inference only (no dropout in kernel)
            output = peri_attn_triton_forward(disp, val, self, delta)
        else:
            # PyTorch path — training + CPU/MPS fallback
            output = self._pytorch_forward(disp, val, delta, x.device)

        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        output = self.resid_dropout(self.c_proj(output))
        return output

    def _pytorch_forward(self, disp, val, delta, device):
        B, nh, T, bd = disp.shape
        hs = val.shape[-1]

        disp_win = self._build_windows(disp, delta)
        strain = disp_win - disp.unsqueeze(3)

        rel_ids = torch.arange(delta, device=device)
        rel_emb = self.rel_pos_emb(rel_ids)
        pos_feat = self.pos_proj(rel_emb)

        fused = self.strain_fused(strain)
        bond_feats, damage_feats = fused.chunk(2, dim=-1)

        bond_logits = self.bond_out(F.gelu(bond_feats + pos_feat)).squeeze(-1)
        damage = torch.sigmoid(self.damage_out(F.gelu(damage_feats)).squeeze(-1))
        bond_logits = bond_logits - damage * 10.0

        cmask = self._causal_mask(T, delta, device)
        bond_logits = bond_logits.masked_fill(~cmask.unsqueeze(0).unsqueeze(0), float('-inf'))

        weights = F.softmax(bond_logits, dim=-1)
        weights = self.attn_dropout(weights)

        output = torch.zeros(B, nh, T, hs, device=device, dtype=val.dtype)
        val_padded = F.pad(val, (0, 0, delta - 1, 0))
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
        disp_win = self._build_windows(disp, delta)
        strain = disp_win - disp.unsqueeze(3)

        fused = self.strain_fused(strain)
        _, damage_feats = fused.chunk(2, dim=-1)
        damage = torch.sigmoid(self.damage_out(F.gelu(damage_feats)).squeeze(-1))
        cmask = self._causal_mask(T, delta, x.device).float()
        valid_count = cmask.sum()
        mean_damage = (damage * cmask.unsqueeze(0).unsqueeze(0)).sum() / (
            valid_count * B * nh + 1e-8)
        return mean_damage.item()
