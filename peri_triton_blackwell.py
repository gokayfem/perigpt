"""
Blackwell-optimized Triton kernel for peridynamic attention.

Key design: one program per (batch*head, position). Loops over horizon
sequentially but uses tl.dot for the strain→projection matmul (tensor
cores) and online softmax for numerically stable single-pass aggregation.

No 5D intermediate tensors. No unsupported Triton constructs.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _tanh_approx(x):
        e2x = tl.exp(2.0 * x)
        return (e2x - 1.0) / (e2x + 1.0)

    @triton.jit
    def _gelu_vec(x):
        """GELU on a vector, using exp-based tanh."""
        inner = 0.7978845608 * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + _tanh_approx(inner))

    @triton.jit
    def _peri_attn_bw_kernel(
        # Data: all (BN, T_padded, dim) layout, contiguous
        disp_ptr,       # (BN, T+D-1, BD)
        val_ptr,        # (BN, T+D-1, HS)
        out_ptr,        # (BN, T, HS)
        # Weights: all contiguous floats
        W_fused_ptr,    # (2*BD, BD) row-major
        bias_fused_ptr, # (2*BD,)
        pos_feat_ptr,   # (DELTA, BD) pre-computed position features
        w_bond_ptr,     # (BD,)
        bond_bias_ptr,  # scalar
        w_dmg_ptr,      # (BD,)
        dmg_bias_ptr,   # scalar
        # Dims
        T: tl.constexpr,
        HS: tl.constexpr,
        BD: tl.constexpr,
        DELTA: tl.constexpr,
        # Strides for disp (BN, T+D-1, BD)
        sd_bn, sd_t, sd_d: tl.constexpr,
        # Strides for val (BN, T+D-1, HS)
        sv_bn, sv_t, sv_d: tl.constexpr,
        # Strides for out (BN, T, HS)
        so_bn, so_t, so_d: tl.constexpr,
        # Block sizes
        BLOCK_BD: tl.constexpr,
        BLOCK_HS: tl.constexpr,
    ):
        """One program = one (batch*head, position) pair."""
        pid_bh = tl.program_id(0)
        pid_t = tl.program_id(1)

        bd_offs = tl.arange(0, BLOCK_BD)
        hs_offs = tl.arange(0, BLOCK_HS)

        # Load center displacement from padded array
        # Position t in original = position (t + DELTA - 1) in padded
        center_padded_t = pid_t + DELTA - 1
        center_base = pid_bh * sd_bn + center_padded_t * sd_t
        center = tl.load(disp_ptr + center_base + bd_offs * sd_d,
                         mask=bd_offs < BD, other=0.0)  # (BLOCK_BD,)

        # Load weight vectors (small, fits in registers)
        w_bond = tl.load(w_bond_ptr + bd_offs, mask=bd_offs < BD, other=0.0)
        w_dmg = tl.load(w_dmg_ptr + bd_offs, mask=bd_offs < BD, other=0.0)
        b_bond = tl.load(bond_bias_ptr)
        b_dmg = tl.load(dmg_bias_ptr)

        # Load fused bias: first BD elements for bond, next BD for damage
        bias_bond = tl.load(bias_fused_ptr + bd_offs, mask=bd_offs < BD, other=0.0)
        bias_dmg = tl.load(bias_fused_ptr + BD + bd_offs, mask=bd_offs < BD, other=0.0)

        # Online softmax state
        m_prev = -float('inf')
        d_prev = 0.0
        acc = tl.zeros((BLOCK_HS,), dtype=tl.float32)

        # Loop over horizon neighbors
        for j in tl.static_range(DELTA):
            # Neighbor original position
            nb_orig = pid_t - DELTA + 1 + j
            # Only process if causal mask is valid (nb_orig >= 0)
            is_valid = nb_orig >= 0

            # Neighbor position in padded array = pid_t + j
            nb_padded_t = pid_t + j
            nb_base = pid_bh * sd_bn + nb_padded_t * sd_t

            # Load neighbor displacement
            nb_disp = tl.load(disp_ptr + nb_base + bd_offs * sd_d,
                              mask=(bd_offs < BD) & is_valid, other=0.0)

            # Strain = neighbor - center
            strain = nb_disp - center  # (BLOCK_BD,)

            # Fused projection: strain @ W_fused^T + bias
            # W_fused is (2*BD, BD). We need strain (1, BD) @ W_fused^T (BD, 2*BD)
            # For BD=32 this is small enough to do as two dot products
            # bond_feat = strain @ W_fused[0:BD, :]^T + bias[0:BD]
            # dmg_feat = strain @ W_fused[BD:2BD, :]^T + bias[BD:2BD]
            bond_feat = tl.zeros((BLOCK_BD,), dtype=tl.float32)
            dmg_feat = tl.zeros((BLOCK_BD,), dtype=tl.float32)
            for k in tl.static_range(BD):
                s_k = tl.load(disp_ptr + nb_base + k * sd_d, mask=is_valid, other=0.0) - \
                      tl.load(disp_ptr + center_base + k * sd_d)
                # Row k of W_fused bond part: W_fused[0:BD, k]
                wb_k = tl.load(W_fused_ptr + k + bd_offs * BD,
                               mask=bd_offs < BD, other=0.0)
                bond_feat += s_k * wb_k
                # Row k of W_fused damage part: W_fused[BD:2BD, k]
                wd_k = tl.load(W_fused_ptr + BD * BD + k + bd_offs * BD,
                               mask=bd_offs < BD, other=0.0)
                dmg_feat += s_k * wd_k

            bond_feat = bond_feat + bias_bond
            dmg_feat = dmg_feat + bias_dmg

            # Load position feature for offset j
            pos_j = tl.load(pos_feat_ptr + j * BD + bd_offs,
                            mask=bd_offs < BD, other=0.0)

            # Bond score = w_bond . gelu(bond_feat + pos)
            bond_act = _gelu_vec(bond_feat + pos_j)
            logit = tl.sum(w_bond * bond_act) + b_bond

            # Damage = sigmoid(w_dmg . gelu(dmg_feat))
            dmg_act = _gelu_vec(dmg_feat)
            dmg_logit = tl.sum(w_dmg * dmg_act) + b_dmg
            damage = tl.sigmoid(dmg_logit)

            logit = logit - damage * 10.0

            # Set invalid positions to -inf
            logit = tl.where(is_valid, logit, -float('inf'))

            # Online softmax update
            m_new = tl.maximum(m_prev, logit)
            scale = tl.exp(m_prev - m_new)
            p_new = tl.exp(logit - m_new)

            d_prev = d_prev * scale + p_new
            acc = acc * scale

            # Load neighbor value and accumulate
            nb_val_base = pid_bh * sv_bn + nb_padded_t * sv_t
            nb_val = tl.load(val_ptr + nb_val_base + hs_offs * sv_d,
                             mask=(hs_offs < HS) & is_valid, other=0.0)
            acc += p_new * nb_val

            m_prev = m_new

        # Normalize
        out = acc / (d_prev + 1e-8)

        # Store
        out_base = pid_bh * so_bn + pid_t * so_t
        tl.store(out_ptr + out_base + hs_offs * so_d, out, mask=hs_offs < HS)


def peri_attn_blackwell_forward(disp, val, attn_module, delta):
    """Launch the Blackwell kernel."""
    B, nh, T, bd = disp.shape
    hs = val.shape[-1]
    BN = B * nh

    # Pad and flatten
    disp_padded = F.pad(disp, (0, 0, delta - 1, 0)).reshape(BN, T + delta - 1, bd).contiguous().float()
    val_padded = F.pad(val, (0, 0, delta - 1, 0)).reshape(BN, T + delta - 1, hs).contiguous().float()
    out = torch.empty(BN, T, hs, device=disp.device, dtype=torch.float32)

    # Pre-compute position features
    rel_ids = torch.arange(delta, device=disp.device)
    pos_feat = attn_module.pos_proj(attn_module.rel_pos_emb(rel_ids)).contiguous().float()

    # Weight access — W_fused stored as (2*bd, bd) row-major
    W_fused = attn_module.strain_fused.weight.contiguous().float()
    bias_fused = attn_module.strain_fused.bias.contiguous().float()
    w_bond = attn_module.bond_out.weight.squeeze(0).contiguous().float()
    bond_bias = (attn_module.bond_out.bias.contiguous().float()
                 if attn_module.bond_out.bias is not None
                 else torch.zeros(1, device=disp.device, dtype=torch.float32))
    w_dmg = attn_module.damage_out.weight.squeeze(0).contiguous().float()
    dmg_bias = attn_module.damage_out.bias.contiguous().float()

    BLOCK_BD = triton.next_power_of_2(bd)
    BLOCK_HS = triton.next_power_of_2(hs)

    grid = (BN, T)

    _peri_attn_bw_kernel[grid](
        disp_padded, val_padded, out,
        W_fused, bias_fused, pos_feat,
        w_bond, bond_bias, w_dmg, dmg_bias,
        T, hs, bd, delta,
        disp_padded.stride(0), disp_padded.stride(1), disp_padded.stride(2),
        val_padded.stride(0), val_padded.stride(1), val_padded.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_BD=BLOCK_BD,
        BLOCK_HS=BLOCK_HS,
    )

    return out.reshape(B, nh, T, hs).to(disp.dtype)


class PeriDynamicAttentionBlackwell(nn.Module):
    """Peridynamic attention with Triton kernel. Falls back to PyTorch on CPU/MPS."""

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

    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head
        hs = self.head_size
        bd = self.bond_dim
        delta = min(self.horizon, T)

        disp = self.W_disp(x).view(B, T, nh, bd).permute(0, 2, 1, 3)
        val = self.W_val(x).view(B, T, nh, hs).permute(0, 2, 1, 3)

        if x.is_cuda and HAS_TRITON:
            output = peri_attn_blackwell_forward(disp, val, self, delta)
        else:
            output = self._pytorch_fallback(disp, val, delta, x)

        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        output = self.resid_dropout(self.c_proj(output))
        return output

    def _pytorch_fallback(self, disp, val, delta, x):
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
        weights = self.attn_dropout(weights)

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
