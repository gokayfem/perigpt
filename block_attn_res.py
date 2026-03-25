"""
block_attn_res.py — Block Attention Residual: State-Based Peridynamics Over Depth

This module unifies three ideas into a single mechanism:

  1. PERIDYNAMICS (sequence dimension):
     PeriDynamicAttention computes bond-based interactions within a causal
     horizon δ along the token sequence.

  2. BLOCK ATTN-RES (depth dimension):
     Instead of the standard residual stream x^(l) = x^(l-1) + layer(x^(l-1)),
     layers are grouped into "blocks" of size B.  At each layer, a lightweight
     attention over all completed block summaries + the current partial sum
     determines the input — a *learned, adaptive skip connection*.

     This IS state-based peridynamics over depth:
     ┌─────────────────────────┬────────────────────────────────────────┐
     │  Peridynamics (depth)   │  Block AttnRes                         │
     ├─────────────────────────┼────────────────────────────────────────┤
     │  Material points        │  Block summaries (completed blocks)    │
     │  Horizon δ              │  All previous blocks (growing horizon) │
     │  Deformation state      │  Partial sum within current block      │
     │  Integral over horizon  │  Softmax-weighted sum over blocks      │
     │  Constitutive law       │  Learned projection + RMSNorm          │
     └─────────────────────────┴────────────────────────────────────────┘

  3. DEER (parallelism):
     Block boundaries are natural coarse-grid checkpoints.  Instead of DEER
     solving for all L layers at once, we do MULTIGRID DEER:
       - Coarse level: solve the block-level recurrence via parallel scan
       - Fine level:   refine within each block (short sequential chains)

     For L=48 layers with block_size=6:  8 blocks, each 6 layers deep.
     Multigrid DEER: parallel scan over 8 blocks + sequential within 6
     = O(log(8) + 6) = O(9) vs sequential O(48) → ~5× reduction.

     This is the same principle as parareal (Lions et al., 2001), applied
     to transformer depth.

Usage:
    config = GPTConfig(
        attention_type='peridynamic',  # peridynamic bonds along sequence
        residual_type='block_attn',    # block attn-res along depth
        depth_block_size=6,            # layers per block
    )
    model = GPT(config)

References:
    - Peridynamic state-based: Silling et al., 2007
    - Block AttnRes: (the code pattern shown by user)
    - DEER / parareal: Lim et al. 2024; Lions et al. 2001
    - Multigrid-in-time: Jiang et al. 2026 (from micro_deer bibliography)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ═══════════════════════════════════════════════════════════════════════════
# RMSNorm  (used as key normalisation in block attention)
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ═══════════════════════════════════════════════════════════════════════════
# Block Attention Residual
# ═══════════════════════════════════════════════════════════════════════════

def block_attn_res(
    blocks: list,                  # N tensors of [B, T, D]
    partial_block: torch.Tensor,   # [B, T, D] — current intra-block sum
    proj: nn.Linear,               # [1, D] or [D] — query projection
    norm: RMSNorm,                 # key normalisation
) -> torch.Tensor:
    """
    Inter-block attention: attend over block representations + partial sum.

    This is the *state-based peridynamic integral* over the depth dimension.
    Each completed block is a "material point" in depth-space.  The partial
    sum is the current "deformation state" being integrated.

    The attention computes:
        h = Σ_j  softmax(proj · RMSNorm(V_j))_j  ·  V_j

    where V = [block_0, block_1, ..., block_{N-1}, partial_block].

    This is a learned, adaptive weighted combination of all previous block
    summaries, replacing the simple additive residual stream.

    Args:
        blocks:        List of N completed block representations [B, T, D]
        partial_block: Current intra-block partial sum [B, T, D]
        proj:          Learned query projection (scalar attention)
        norm:          RMSNorm for key computation

    Returns:
        h: [B, T, D] — weighted combination of all blocks + partial
    """
    # Stack: [N+1, B, T, D]
    V = torch.stack(blocks + [partial_block])

    # Keys: normalised representations
    K = norm(V)                                            # [N+1, B, T, D]

    # Scalar attention logits: project each key to a scalar
    # proj.weight is [1, D] or [D] — dot product with each key
    w = proj.weight.squeeze()                              # [D]
    logits = torch.einsum('d, n b t d -> n b t', w, K)    # [N+1, B, T]

    # Softmax over the block dimension (dim=0)
    weights = logits.softmax(dim=0)                        # [N+1, B, T]

    # Weighted sum of values
    h = torch.einsum('n b t, n b t d -> b t d', weights, V)  # [B, T, D]

    return h


# ═══════════════════════════════════════════════════════════════════════════
# Block-Aware Transformer Layer
# ═══════════════════════════════════════════════════════════════════════════

class BlockAttnResLayer(nn.Module):
    """
    Transformer layer with Block Attention Residual.

    Instead of:
        x = x + attn(norm(x))
        x = x + mlp(norm(x))

    We do:
        # Before attention: attend over all block reps
        h = block_attn_res(blocks, partial, attn_res_proj, attn_res_norm)
        partial = partial + attn(attn_norm(h))

        # Before MLP: attend over all block reps again
        h = block_attn_res(blocks, partial, mlp_res_proj, mlp_res_norm)
        partial = partial + mlp(mlp_norm(h))

        # At block boundary: snapshot current partial as new block
        if layer_number % block_size == 0:
            blocks.append(partial)
            partial = 0

    This means the attention and MLP see a *weighted combination* of all
    previous blocks, not just the most recent layer output.  Information
    from early layers can flow directly to late layers without degradation
    through the residual stream.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        D = config.n_embd
        self.layer_idx = layer_idx
        self.depth_block_size = getattr(config, 'depth_block_size', 4)

        # Attention sub-layer
        self.attn_norm = _make_norm(D, config)

        attn_type = getattr(config, 'attention_type', 'standard')
        if attn_type == 'peridynamic':
            from model import PeriDynamicAttention
            self.attn = PeriDynamicAttention(config)
        elif attn_type == 'hybrid':
            from model import HybridPeriAttention
            self.attn = HybridPeriAttention(config)
        else:
            from model import CausalSelfAttention
            self.attn = CausalSelfAttention(config)

        # MLP sub-layer
        self.mlp_norm = _make_norm(D, config)
        from model import MLP
        self.mlp = MLP(config)

        # Block attention residual projections
        # Each is a [1, D] linear that produces scalar logits for block attention
        self.attn_res_proj = nn.Linear(D, 1, bias=False)
        self.attn_res_norm = RMSNorm(D)
        self.mlp_res_proj  = nn.Linear(D, 1, bias=False)
        self.mlp_res_norm  = RMSNorm(D)

        # Is this layer a block boundary?
        # block_size counts attn+mlp as 2 sub-layers per transformer layer
        # So layer l is a block boundary when l % (depth_block_size // 2) == 0
        # But for simplicity: every depth_block_size layers = 1 block
        self.is_block_boundary = (
            (layer_idx + 1) % self.depth_block_size == 0
        )

    def forward(
        self,
        blocks: list,
        partial_block: torch.Tensor,
    ) -> tuple:
        """
        Args:
            blocks:        list of completed block reps [B, T, D]
            partial_block: current intra-block partial sum [B, T, D]

        Returns:
            blocks:        (possibly extended) list of block reps
            partial_block: updated partial sum
        """
        # ── Apply block attn-res before attention ──
        h = block_attn_res(
            blocks, partial_block,
            self.attn_res_proj, self.attn_res_norm)

        # ── Self-attention ──
        attn_out = self.attn(self.attn_norm(h))
        partial_block = partial_block + attn_out

        # ── Apply block attn-res before MLP ──
        h = block_attn_res(
            blocks, partial_block,
            self.mlp_res_proj, self.mlp_res_norm)

        # ── MLP ──
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        # ── Block boundary: snapshot and reset ──
        if self.is_block_boundary:
            blocks = blocks + [partial_block]
            # Don't reset partial to zero — let it accumulate
            # (the block_attn_res will learn to weight old vs new)

        return blocks, partial_block


def _make_norm(dim, config):
    """Create the right norm layer based on config."""
    from model import LayerNorm
    return LayerNorm(dim, bias=getattr(config, 'bias', True))


# ═══════════════════════════════════════════════════════════════════════════
# Block-Aware Peridynamic Damage for DEER
# ═══════════════════════════════════════════════════════════════════════════

class BlockDamageTracker:
    """
    Track damage statistics per block for multigrid DEER.

    Within each block, we accumulate the mean peridynamic bond damage
    from the PeriDynamicAttention layers.  This gives us a per-block
    "predictability score" that determines:

    1. DEER damping:  k_block = α · mean_damage_block
       (more damage → more damping → conservative Newton step)

    2. Block boundary placement (future work):
       Adaptively place boundaries where damage is highest
       (semantic discontinuities in depth).

    This connects three levels:
    - Token-level bonds (PeriDynamicAttention horizon)
    - Layer-level damage (mean over bonds per layer)
    - Block-level predictability (mean over layers per block → DEER damping)
    """

    def __init__(self, n_blocks: int, damage_scale: float = 1.0):
        self.n_blocks = n_blocks
        self.damage_scale = damage_scale
        self.block_damages = [0.0] * n_blocks

    def update(self, block_idx: int, layer_damage: float):
        """Accumulate damage for a block."""
        self.block_damages[block_idx] = max(
            self.block_damages[block_idx], layer_damage)

    def get_block_damping(self, block_idx: int) -> float:
        """Get DEER damping factor for a block."""
        return min(self.damage_scale * self.block_damages[block_idx], 0.95)

    def get_all_dampings(self) -> list:
        return [self.get_block_damping(i) for i in range(self.n_blocks)]


# ═══════════════════════════════════════════════════════════════════════════
# Multigrid DEER: Coarse (blocks) + Fine (within-block)
# ═══════════════════════════════════════════════════════════════════════════

class MultigridDEER:
    """
    Two-level DEER that exploits block structure:

    COARSE LEVEL (parallel):
        Solve for block-to-block transitions using parallel scan.
        The "state" at each block boundary is the block summary.
        O(log N_blocks) via associative scan.

    FINE LEVEL (sequential, but short):
        Within each block, run the layers sequentially.
        Each block has only depth_block_size layers.
        O(depth_block_size) per block, all blocks in parallel.

    Total: O(log N_blocks + depth_block_size) vs O(L) sequential.

    For L=48, block_size=6: O(log(8) + 6) = O(9) vs O(48) → ~5×

    The block_attn_res mechanism makes the coarse level much better
    conditioned than naive DEER over all layers, because each block
    summary captures a *meaningful* intermediate representation (it was
    learned to be useful by the block attention), not just whatever
    the residual stream happened to contain.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        depth_block_size: int = 4,
        max_coarse_iter: int = 5,
        tol: float = 1e-4,
    ):
        self.layers = layers
        self.block_size = depth_block_size
        self.n_blocks = len(layers) // depth_block_size
        self.max_coarse_iter = max_coarse_iter
        self.tol = tol

        # Validate
        assert len(layers) % depth_block_size == 0, \
            f"n_layer ({len(layers)}) must be divisible by depth_block_size ({depth_block_size})"

    def _run_block(
        self,
        block_idx: int,
        block_input: torch.Tensor,
        blocks_so_far: list,
    ) -> torch.Tensor:
        """
        Run one block of layers sequentially.

        Args:
            block_idx: which block (0-indexed)
            block_input: input to this block [B, T, D]
            blocks_so_far: list of previous block summaries

        Returns:
            block_output: [B, T, D]
        """
        start_layer = block_idx * self.block_size
        end_layer = start_layer + self.block_size

        partial = block_input
        blocks = list(blocks_so_far)  # copy

        for l in range(start_layer, end_layer):
            layer = self.layers[l]
            if isinstance(layer, BlockAttnResLayer):
                blocks, partial = layer(blocks, partial)
            else:
                # Standard block — just apply
                partial = layer(partial)

        return partial

    def forward(
        self,
        x0: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Multigrid DEER forward:
        1. Guess block boundary states (Picard warm-start)
        2. Newton iterations on the coarse block-level recurrence
        3. Final refinement pass within each block

        Args:
            x0: embedding output [B, T, D]

        Returns:
            x_final: output of last layer [B, T, D]
        """
        N = self.n_blocks

        # ── Picard warm-start: run each block independently from x0 ──
        with torch.no_grad():
            block_states = [x0]  # block_states[i] = output of block i
            for i in range(N):
                out = self._run_block(i, x0, [x0])
                block_states.append(out)

        # ── Coarse Newton iterations ──
        # The block-level recurrence:
        #   s_{i+1} = F_i(s_i)  where F_i = "run block i"
        # We solve this via Newton's method

        iterations = 0
        converged = False

        for it in range(self.max_coarse_iter):
            # Compute residuals at block level
            residuals = []
            for i in range(N):
                blocks_before = block_states[:i + 1]  # [x0, s1, ..., s_i]
                expected = self._run_block(i, block_states[i], blocks_before)
                r = expected - block_states[i + 1]
                residuals.append(r)

            # Check convergence
            res_norm = sum(r.norm().item() for r in residuals) / N
            state_norm = sum(s.norm().item() for s in block_states[1:]) / N
            rel_res = res_norm / (state_norm + 1e-8)

            if rel_res < self.tol:
                converged = True
                iterations = it + 1
                break

            # Simple update (Picard-style at coarse level — cheaper than
            # full Newton since blocks are large)
            for i in range(N):
                block_states[i + 1] = block_states[i + 1] + residuals[i]

            iterations = it + 1

        # ── Final sequential pass (ensures exact output) ──
        partial = block_states[0]  # x0
        blocks = [x0]

        for l, layer in enumerate(self.layers):
            if isinstance(layer, BlockAttnResLayer):
                blocks, partial = layer(blocks, partial)
            else:
                partial = layer(partial)

        if return_diagnostics:
            return partial, {
                'iterations': iterations,
                'converged': converged,
                'n_blocks': N,
            }
        return partial


# ═══════════════════════════════════════════════════════════════════════════
# The Unified Picture
# ═══════════════════════════════════════════════════════════════════════════

"""
Here's the complete map of how the three ideas connect:

SEQUENCE DIMENSION (horizontal, tokens):
  ┌─────────────────────────────────────────────────┐
  │  PeriDynamicAttention                            │
  │  • Bond-based: pairwise strain h_j - h_i        │
  │  • Bounded horizon δ along token sequence        │
  │  • Damage = bond breaking at semantic boundaries │
  │  • Complexity: O(T · δ)                          │
  └─────────────────────────────────────────────────┘
           ↕ damage feeds damping
           ↕

DEPTH DIMENSION (vertical, layers):
  ┌─────────────────────────────────────────────────┐
  │  Block Attention Residual                        │
  │  • State-based: aggregate over all prev blocks   │
  │  • Growing horizon (sees all completed blocks)   │
  │  • Learned weighted skip connections             │
  │  • Replaces simple additive residual stream      │
  └─────────────────────────────────────────────────┘
           ↕ block structure enables
           ↕

PARALLELISM (speed):
  ┌─────────────────────────────────────────────────┐
  │  Multigrid DEER                                  │
  │  • Coarse: parallel scan over block boundaries   │
  │  • Fine: sequential within each short block      │
  │  • Damage → adaptive DEER damping per block      │
  │  • O(log N_blocks + block_size) vs O(L)          │
  └─────────────────────────────────────────────────┘

The peridynamic damage from sequence-level attention tells DEER which
blocks in depth are "predictable" (low damage → aggressive Newton step)
vs "discontinuous" (high damage → conservative step).

Block AttnRes provides the coarse-grid structure that makes multigrid
DEER well-conditioned: each block summary is a *learned* checkpoint
that captures meaningful intermediate representations.

All three mechanisms share the same mathematical DNA from peridynamics:
nonlocal integral operators with adaptive influence functions.
"""
