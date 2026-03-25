"""
nanoPeriGPT — nanoGPT with Peridynamic Attention + DEER Layer-Parallelism

Drop-in replacement for nanoGPT's model.py.  Adds PeriDynamicAttention as an
alternative to CausalSelfAttention, inspired by bond-based / state-based
peridynamics in computational mechanics.  Optionally parallelises the
sequential layer-by-layer computation using DEER (Newton's method + parallel
associative scan), with peridynamic damage as an adaptive damping signal.

Key peridynamic ideas mapped to sequence modeling:
  ┌─────────────────────────┬──────────────────────────────────────┐
  │  Peridynamics           │  This module                         │
  ├─────────────────────────┼──────────────────────────────────────┤
  │  Material point         │  Token embedding                     │
  │  Horizon δ              │  Causal sliding window of size δ     │
  │  Bond strain  x'-x      │  h_j - h_i  (relative difference)   │
  │  Micro-modulus / force  │  Learned bond score function f_θ     │
  │  Bond damage            │  Adaptive sparsity via learned       │
  │                         │  critical-strain threshold           │
  │  Deformation state      │  Neighborhood aggregation (state)    │
  │  Integral over horizon  │  Weighted sum over window            │
  └─────────────────────────┴──────────────────────────────────────┘

DEER layer-parallelism (via deer_parallel.py):
  - Replaces sequential block_0 → block_1 → ... → block_L with
    a parallel Newton solve over all layers simultaneously
  - Complexity: O(K · log L) instead of O(L)  (K = Newton iterations)
  - damage-ELK: peridynamic damage drives per-layer adaptive damping

Usage:
    # Standard attention (identical to nanoGPT)
    config = GPTConfig(attention_type='standard')
    model = GPT(config)

    # Peridynamic attention
    config = GPTConfig(attention_type='peridynamic', horizon=32)
    model = GPT(config)

    # Peridynamic + DEER layer-parallel inference
    config = GPTConfig(attention_type='peridynamic', horizon=32,
                       deer_enabled=True, deer_method='damage-elk')
    model = GPT(config)

    # Enable DEER on a pre-trained model at inference time
    model.enable_deer(method='quasi-deer', max_iter=10)

References:
  1) nanoGPT — https://github.com/karpathy/nanoGPT
  2) Silling, "Reformulation of elasticity theory for discontinuities
     and long-range forces", J. Mech. Phys. Solids, 2000
  3) Silling et al., "Peridynamic states and constitutive modeling", 2007
  4) Lim et al., "Parallelizing non-linear sequential models" (DEER), 2024
  5) Gonzalez et al., "Towards Scalable and Stable Parallelization" (ELK), 2024
  6) Gonzalez et al., "Predictability Enables Parallelization", 2025
  7) lindermanlab/micro_deer — https://github.com/lindermanlab/micro_deer
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Shared modules (identical to nanoGPT)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# ---------------------------------------------------------------------------
# Standard Causal Self-Attention (unchanged from nanoGPT)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

# ---------------------------------------------------------------------------
# Peridynamic Attention  (the new module)
# ---------------------------------------------------------------------------

class PeriDynamicAttention(nn.Module):
    """
    Peridynamic-inspired causal attention.

    For each token i, instead of computing softmax(Q_i · K^T / √d) over ALL
    previous tokens, we:

      1. Look only within a bounded *horizon* δ  (causal window).
      2. Compute the *strain* (relative difference) between the token and
         each neighbour:  ε_{ij} = disp_j − disp_i
      3. Feed strain + relative-position encoding into a learned *bond force
         function* f_θ that outputs a scalar "bond score" per neighbour.
      4. A *damage function* g_θ produces a scalar damage d ∈ [0,1] from
         the strain; bonds with high damage are suppressed (the peridynamic
         analog of crack formation / semantic discontinuities).
      5. Scores are normalised via softmax over the horizon and used to
         aggregate a *value* representation — the "peridynamic integral".

    Memory-efficiency trick
    -----------------------
    Strain is computed in a *reduced bond dimension* (bond_dim ≈ head_size/4)
    so the (B, nh, T, δ, bond_dim) intermediate is ≈ 4× smaller than if we
    used the full head_size.  Values remain full-dimensional.

    Complexity: O(T · δ)  per head  — linear in T for fixed horizon δ.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head    = config.n_head
        self.n_embd    = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.horizon   = getattr(config, 'horizon', 32)
        self.dropout   = config.dropout

        hs = self.head_size
        # Reduced bond dimension — keeps strain tensors small
        self.bond_dim = max(hs // 4, 16)
        bd = self.bond_dim

        # ---------- projections ----------
        # "Displacement" space  (reduced dim, used for strain)
        self.W_disp = nn.Linear(config.n_embd, config.n_head * bd, bias=config.bias)
        # "Value" space  (full dim, carries information to aggregate)
        self.W_val  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # ---------- relative position embedding ----------
        # Encodes the *reference configuration* geometry within the horizon
        self.rel_pos_emb = nn.Embedding(self.horizon, bd)

        # ---------- bond force / score function  (the "micro-modulus") ----------
        # Maps strain features + positional features → scalar bond score.
        # Factorised to avoid concatenating large tensors:
        #   score = Linear( GELU( strain_proj(strain) + pos_proj(rel_pos) ) )
        self.strain_proj = nn.Linear(bd, bd, bias=config.bias)
        self.pos_proj    = nn.Linear(bd, bd, bias=False)       # no bias; absorbed by strain_proj
        self.bond_out    = nn.Linear(bd, 1, bias=config.bias)

        # ---------- damage function  (bond breaking) ----------
        # Learns when the "semantic strain" between two tokens exceeds a
        # critical threshold → the bond breaks → adaptive sparsity.
        # Outputs a scalar damage ∈ [0, 1] per bond.
        # Always use bias here — the bias init controls starting damage level.
        self.damage_proj = nn.Linear(bd, bd, bias=True)
        self.damage_out  = nn.Linear(bd, 1, bias=True)
        # Initialise so bonds start mostly intact:  sigmoid(−3) ≈ 0.047
        nn.init.constant_(self.damage_out.bias, -3.0)

        # ---------- output projection ----------
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    # ---- helpers -----------------------------------------------------------

    def _build_windows(self, tensor, delta):
        """Left-pad `tensor` by δ−1 on the time axis, then unfold to get
        causal sliding windows of width δ.

        Input:  (B, nh, T, D)
        Output: (B, nh, T, δ, D)   — a *view* (no copy) when possible.
        """
        padded = F.pad(tensor, (0, 0, delta - 1, 0))          # (B, nh, T+δ−1, D)
        win = padded.unfold(2, delta, 1)                       # (B, nh, T, D, δ)
        return win.permute(0, 1, 2, 4, 3)                     # (B, nh, T, δ, D)

    def _causal_mask(self, T, delta, device):
        """Boolean mask: True where the neighbour position is non-negative.

        For token at position t, the window covers original positions
        [t−δ+1, …, t].  Positions < 0 are zero-padded and must be masked.
        """
        t_idx = torch.arange(T, device=device).unsqueeze(1)       # (T, 1)
        w_idx = torch.arange(delta, device=device).unsqueeze(0)   # (1, δ)
        return (t_idx - delta + 1 + w_idx) >= 0                   # (T, δ)

    # ---- forward -----------------------------------------------------------

    def forward(self, x):
        B, T, C = x.size()
        nh  = self.n_head
        hs  = self.head_size
        bd  = self.bond_dim
        delta = min(self.horizon, T)

        # ---- project -------------------------------------------------------
        disp = self.W_disp(x).view(B, T, nh, bd).permute(0, 2, 1, 3)   # (B, nh, T, bd)
        val  = self.W_val(x).view(B, T, nh, hs).permute(0, 2, 1, 3)    # (B, nh, T, hs)

        # ---- build causal windows ------------------------------------------
        disp_win = self._build_windows(disp, delta)   # (B, nh, T, δ, bd)
        val_win  = self._build_windows(val,  delta)    # (B, nh, T, δ, hs)

        # causal mask — (T, δ) bool, True = valid
        cmask = self._causal_mask(T, delta, x.device)  # (T, δ)

        # ---- strain  (the core peridynamic quantity) -----------------------
        #   ε_{ij} = disp_j − disp_i
        # This is the *relative deformation*: the interaction depends on how
        # much the two points have moved relative to each other — NOT on their
        # absolute positions.  This gives translational invariance in feature
        # space, analogous to material-frame indifference in mechanics.
        disp_i = disp.unsqueeze(3)                     # (B, nh, T, 1, bd)
        strain = disp_win - disp_i                     # (B, nh, T, δ, bd)

        # ---- relative position embedding -----------------------------------
        rel_ids = torch.arange(delta, device=x.device)
        rel_emb = self.rel_pos_emb(rel_ids)            # (δ, bd)

        # ---- bond score  (micro-modulus / constitutive law) ----------------
        # Factorised:  score = w^T · GELU( W_s · strain + W_p · rel_pos )
        # Avoids materialising a (B,nh,T,δ,2·bd) concatenation.
        strain_feat = self.strain_proj(strain)          # (B, nh, T, δ, bd)
        pos_feat    = self.pos_proj(rel_emb)            # (δ, bd)  — broadcasts
        bond_act    = F.gelu(strain_feat + pos_feat)    # (B, nh, T, δ, bd)
        bond_logits = self.bond_out(bond_act).squeeze(-1)  # (B, nh, T, δ)

        # ---- damage  (adaptive bond breaking) -----------------------------
        # Damage is a function of the strain magnitude.  High damage → the
        # bond is "broken" → its contribution is suppressed.  This is the
        # peridynamic mechanism for handling discontinuities (cracks).
        # In language: semantic boundaries, topic shifts, clause breaks.
        damage_act  = F.gelu(self.damage_proj(strain))     # (B, nh, T, δ, bd)
        damage      = torch.sigmoid(self.damage_out(damage_act).squeeze(-1))
        # damage ∈ [0, 1];  0 = intact,  1 = fully broken

        # Suppress broken bonds by subtracting a large penalty from their logits
        bond_logits = bond_logits - damage * 10.0

        # Apply causal mask  (padded / future positions → −inf)
        bond_logits = bond_logits.masked_fill(
            ~cmask.unsqueeze(0).unsqueeze(0), float('-inf')
        )

        # ---- normalise via softmax  (peridynamic "volume correction") ------
        weights = F.softmax(bond_logits, dim=-1)        # (B, nh, T, δ)
        weights = self.attn_dropout(weights)

        # ---- peridynamic integral: aggregate values over horizon -----------
        # output_i = ∫_{H(i)} weight_{ij} · value_j  dV_j
        output = torch.einsum('bntd,bntde->bnte', weights, val_win)
        # → (B, nh, T, hs)

        # ---- reshape and project -------------------------------------------
        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        output = self.resid_dropout(self.c_proj(output))
        return output

    def get_damage_stats(self, x):
        """Diagnostic: return per-head mean damage for visualisation."""
        B, T, C = x.size()
        bd    = self.bond_dim
        delta = min(self.horizon, T)
        nh    = self.n_head

        disp     = self.W_disp(x).view(B, T, nh, bd).permute(0, 2, 1, 3)
        disp_win = self._build_windows(disp, delta)
        strain   = disp_win - disp.unsqueeze(3)

        damage_act = F.gelu(self.damage_proj(strain))
        damage     = torch.sigmoid(self.damage_out(damage_act).squeeze(-1))
        cmask      = self._causal_mask(T, delta, x.device).float()

        # mean damage per head, only over valid (non-padded) bonds
        valid_count = cmask.sum()
        mean_damage = (damage * cmask.unsqueeze(0).unsqueeze(0)).sum() / (
            valid_count * B * nh + 1e-8
        )
        return mean_damage.item()

# ---------------------------------------------------------------------------
# Hybrid Attention  (peridynamic local + sparse global anchors)
# ---------------------------------------------------------------------------

class HybridPeriAttention(nn.Module):
    """
    Combines peridynamic local attention (horizon δ) with sparse global
    "tendons" — a small set of anchor tokens that every position can attend
    to, handling genuine long-range dependencies cheaply.

    Complexity: O(T · δ  +  T · A)  where A = n_global_anchors ≪ T.
    """

    def __init__(self, config):
        super().__init__()
        self.peri_attn = PeriDynamicAttention(config)

        # Global anchors: every A-th token becomes a "tendon"
        self.n_anchors = getattr(config, 'n_global_anchors', 8)
        hs = config.n_embd // config.n_head

        # Lightweight cross-attention to anchors
        self.W_q_global = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_k_global = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_v_global = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.n_head    = config.n_head
        self.head_size = hs
        self.mix_gate  = nn.Parameter(torch.tensor(0.0))   # learned local/global mix

    def forward(self, x):
        B, T, C = x.size()
        nh, hs  = self.n_head, self.head_size

        # ---- local peridynamic attention -----------------------------------
        local_out = self.peri_attn(x)                    # (B, T, C)

        # ---- sparse global attention to anchor tokens ----------------------
        # Select anchor positions: evenly spaced, causal
        stride    = max(T // self.n_anchors, 1)
        anchor_idx = torch.arange(0, T, stride, device=x.device)  # (A,)

        q = self.W_q_global(x).view(B, T, nh, hs).permute(0, 2, 1, 3)
        k = self.W_k_global(x[:, anchor_idx]).view(B, -1, nh, hs).permute(0, 2, 1, 3)
        v = self.W_v_global(x[:, anchor_idx]).view(B, -1, nh, hs).permute(0, 2, 1, 3)
        # k, v: (B, nh, A, hs)

        # Causal mask: token t can only attend to anchors at positions ≤ t
        A = anchor_idx.size(0)
        # anchor_idx: (A,),  t_idx: (T,)
        t_idx = torch.arange(T, device=x.device).unsqueeze(1)     # (T, 1)
        g_mask = (anchor_idx.unsqueeze(0) <= t_idx)                # (T, A) bool
        g_mask = g_mask.unsqueeze(0).unsqueeze(0)                  # (1, 1, T, A)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, nh, T, A)
        att = att.masked_fill(~g_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        global_out = (att @ v)                                     # (B, nh, T, hs)
        global_out = global_out.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # ---- gate and combine ----------------------------------------------
        alpha = torch.sigmoid(self.mix_gate)
        return local_out * (1 - alpha) + global_out * alpha

# ---------------------------------------------------------------------------
# Feed-Forward (MLP) — unchanged from nanoGPT
# ---------------------------------------------------------------------------

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        attn_type = getattr(config, 'attention_type', 'standard')
        if attn_type == 'peridynamic':
            self.attn = PeriDynamicAttention(config)
        elif attn_type == 'hybrid':
            self.attn = HybridPeriAttention(config)
        else:
            self.attn = CausalSelfAttention(config)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp   = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304   # GPT-2 vocab_size of 50257, padded to multiple of 64
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768
    dropout:  float = 0.0
    bias:      bool = True

    # --- peridynamic extensions ---
    attention_type:   str = 'standard'   # 'standard' | 'peridynamic' | 'hybrid'
    horizon:          int = 32           # δ — neighbourhood size for peridynamic attn
    n_global_anchors: int = 8            # number of global "tendon" tokens (hybrid only)

    # --- block attention residual (state-based peridynamics over depth) ---
    residual_type:    str = 'standard'   # 'standard' | 'block_attn'
    depth_block_size: int = 4            # layers per block for block_attn residual

    # --- DEER layer-parallel extensions ---
    deer_enabled:     bool = False       # enable DEER-parallel forward pass
    deer_method:      str = 'quasi-deer' # 'deer' | 'quasi-deer' | 'elk' | 'damage-elk'
    deer_max_iter:    int = 10           # max Newton iterations
    deer_tol:       float = 1e-4         # convergence tolerance
    deer_damping:   float = 0.3          # fixed damping for ELK (0=DEER, 1=Jacobi)
    deer_damage_scale: float = 1.0       # damage → damping scaling for damage-elk
    deer_warmstart:   str = 'picard'     # 'zero' | 'picard' | 'copy'

    # --- energy formulation (Phase 7) ---
    energy_lambda:  float = 0.0          # weight of strain energy auxiliary loss (0 = off)

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = self._make_layers(config),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # scaled init for residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # ── DEER layer-parallel engine ──
        self._deer_engine = None
        if getattr(config, 'deer_enabled', False):
            self._init_deer()

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
        if getattr(config, 'residual_type', 'standard') == 'block_attn':
            n_blocks = config.n_layer // config.depth_block_size
            print(f"Block AttnRes: {n_blocks} blocks × {config.depth_block_size} layers")

    @staticmethod
    def _make_layers(config):
        """Create layer stack: standard Blocks or BlockAttnResLayers."""
        residual_type = getattr(config, 'residual_type', 'standard')
        if residual_type == 'block_attn':
            from block_attn_res import BlockAttnResLayer
            return nn.ModuleList([
                BlockAttnResLayer(config, layer_idx=i)
                for i in range(config.n_layer)
            ])
        else:
            return nn.ModuleList([Block(config) for _ in range(config.n_layer)])

    def _init_deer(self):
        """Initialise the DEER layer-parallel forward engine."""
        from deer_parallel import DEERForward, DEERConfig
        cfg = self.config
        deer_cfg = DEERConfig(
            method       = cfg.deer_method,
            max_iter     = cfg.deer_max_iter,
            tol          = cfg.deer_tol,
            damping      = cfg.deer_damping,
            damage_scale = cfg.deer_damage_scale,
            warmstart    = cfg.deer_warmstart,
        )
        self._deer_engine = DEERForward(self.transformer.h, deer_cfg)
        print(f"DEER enabled: method={cfg.deer_method}, max_iter={cfg.deer_max_iter}, "
              f"tol={cfg.deer_tol}")

    def enable_deer(self, **kwargs):
        """Enable DEER at runtime (e.g. for inference after sequential training)."""
        from deer_parallel import DEERForward, DEERConfig
        deer_cfg = DEERConfig(**kwargs)
        self._deer_engine = DEERForward(self.transformer.h, deer_cfg)
        print(f"DEER enabled dynamically: {kwargs}")

    def disable_deer(self):
        """Switch back to sequential forward."""
        self._deer_engine = None
        print("DEER disabled — using sequential forward")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)      # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)      # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # ── Layer computation: sequential or DEER-parallel ──
        if self._deer_engine is not None and not self.training:
            # DEER mode — parallelise layers over depth (inference only)
            x = self._deer_engine(x)
        elif getattr(self.config, 'residual_type', 'standard') == 'block_attn':
            # Block Attention Residual: state-based peridynamics over depth
            from block_attn_res import BlockAttnResLayer
            blocks = [x]         # initial block = token embedding
            partial = x          # partial sum starts from embedding
            for layer in self.transformer.h:
                blocks, partial = layer(blocks, partial)
            x = partial
        else:
            # Standard sequential forward
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def forward_deer(self, idx, targets=None, return_diagnostics=False):
        """
        Explicit DEER-parallel forward (always uses DEER, even during training).

        Useful for:
          - Benchmarking DEER vs sequential
          - Research into DEER convergence during training
          - Profiling layer-parallel latency

        Note: gradients through DEER are approximate (diagonal Jacobian).
        For exact training, use standard forward() which is sequential.
        """
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x0 = self.transformer.drop(tok_emb + pos_emb)

        # Ensure DEER engine exists
        if self._deer_engine is None:
            self._init_deer() if getattr(self.config, 'deer_enabled', False) \
                else self.enable_deer(method='quasi-deer', max_iter=10)

        x, diag = self._deer_engine(x0, return_diagnostics=True)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if return_diagnostics:
            return logits, loss, diag
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Load pretrained GPT-2 weights (standard attention only)."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        config_args['attention_type'] = 'standard'  # pretrained = standard attn
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model  = GPT(config)
        sd     = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf  = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf     = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay   = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, "
              f"with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, "
              f"with {num_nodecay:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                       betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N   = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token  = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter   = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved   = flops_per_iter * (1.0 / dt)
        flops_promised   = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                           else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # ---- diagnostic: get damage stats across all layers --------------------
    def get_all_damage_stats(self, x):
        """Run a forward pass and collect mean damage per layer.
        Useful for monitoring whether bonds are learning to break."""
        device = x.device
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        h = self.transformer.drop(
            self.transformer.wte(x) + self.transformer.wpe(pos))

        stats = {}
        for i, block in enumerate(self.transformer.h):
            normed = block.ln_1(h)
            if isinstance(block.attn, PeriDynamicAttention):
                stats[f'layer_{i}'] = block.attn.get_damage_stats(normed)
            elif isinstance(block.attn, HybridPeriAttention):
                stats[f'layer_{i}'] = block.attn.peri_attn.get_damage_stats(normed)
            h = block(h)
        return stats
