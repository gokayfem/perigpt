# PeriGPT: Peridynamic Attention for Heterogeneous Sequence Modeling

**Attention mechanisms derived from nonlocal continuum mechanics that discover structural boundaries through damage mechanics.**

PeriGPT maps peridynamic theory — a nonlocal reformulation of continuum mechanics designed for fracture and discontinuities — onto transformer attention. The result is an attention mechanism where "cracks" (suppressed attention bonds) emerge at domain boundaries in heterogeneous training data, without any explicit supervision.

---

## Key Results

### 1. Peridynamic attention outperforms standard attention on heterogeneous data

On homogeneous text (Shakespeare), peridynamic attention underperforms standard attention — there are no discontinuities to exploit. On heterogeneous multi-domain data (Shakespeare + Python + JSON + Federalist Papers interleaved), the ranking **flips**:

| | Shakespeare (homogeneous) | Mixed-domain (heterogeneous) |
|---|---|---|
| Best peridynamic vs baseline | +0.0913 worse | **-0.0095 better** |
| Block attn-res vs baseline | **-0.0922 better** | +0.0223 worse |

The inductive bias matters, and its value depends on data structure — exactly as the peridynamic theory predicts.

**Shakespeare results (val_loss, 5000 iters):**

| Experiment | val_loss | train_loss | Params |
|---|---|---|---|
| block_attn_res | **1.6001** | 0.7227 | 10.66M |
| baseline (standard) | 1.6987 | 0.6300 | 10.65M |
| peri + block_attn_res | 1.7487 | 0.5856 | 9.12M |
| hybrid (peri + global) | 1.7919 | 0.6011 | 11.76M |
| peridynamic h=64 | 1.7912 | 0.5496 | 9.11M |
| peridynamic h=128 | 1.8056 | 0.5294 | 9.11M |
| peridynamic h=32 | 1.8327 | 0.5248 | 9.11M |
| peridynamic h=16 | 1.8267 | 0.5382 | 9.11M |

**Mixed-domain results (val_loss, 5000 iters):**

| Experiment | val_loss | train_loss | Params |
|---|---|---|---|
| **peridynamic h=128** | **0.8489** | 0.7096 | 9.13M |
| peridynamic h=64 | 0.8515 | 0.7072 | 9.13M |
| peridynamic h=32 | 0.8542 | 0.7178 | 9.13M |
| hybrid (peri + global) | 0.8558 | 0.7253 | 11.76M |
| baseline (standard) | 0.8584 | 0.7438 | 10.65M |
| peri + block_attn_res | 0.8602 | 0.7206 | 9.12M |
| block_attn_res | 0.8807 | 0.7833 | 10.66M |

### 2. Damage learns to break at domain boundaries

The most important result. Without any labels or explicit boundary signal, the peridynamic damage mechanism learns to assign higher damage at domain transitions:

```
Mean damage at domain BOUNDARIES:  0.4204 (+-0.1593)
Mean damage at domain INTERIOR:    0.3594 (+-0.0068)
Ratio (boundary / interior):       1.17x
```

All 6 transformer layers detect boundaries. Interior damage has extremely low variance (+-0.0068), indicating stable within-domain behavior. Boundary variance is high (+-0.1593), reflecting that some transitions are harder to detect than others (Shakespeare-to-JSON is sharp; Shakespeare-to-Federalist-Papers is subtle).

This is **emergent structural segmentation** — the model discovers domain boundaries as a byproduct of next-token prediction, mediated by the physics of the damage mechanism.

---

## Theoretical Framework

### The Peridynamic-Attention Isomorphism

Peridynamics (Silling, 2000) replaces the local PDE formulation of continuum mechanics with a nonlocal integral equation:

```
rho * u_tt(x, t) = integral_H f(u(x') - u(x), x' - x) dV' + b(x, t)
```

Each material point interacts with neighbors within a horizon delta through bond force functions that depend on **relative deformation** (strain). When strain exceeds a critical threshold, bonds **break** — this is how cracks propagate.

The mapping to attention:

| Peridynamics | PeriGPT |
|---|---|
| Material point | Token embedding |
| Horizon delta | Causal sliding window of size delta |
| Bond strain (x' - x) | Relative feature difference (h_j - h_i) |
| Micro-modulus / force function | Learned bond score function f_theta |
| Bond damage | Adaptive sparsity via learned critical-strain threshold |
| Deformation state (state-based) | Neighborhood aggregation before scoring |
| Strain energy density | Auxiliary regularization loss |
| Integral over horizon | Softmax-weighted sum over window |

### Bond-Based vs State-Based

**Bond-based** (implemented as `PeriDynamicAttention`):
```
score(i, j) = f_theta(h_j - h_i, pos_j - pos_i)
```
Each bond is independent. Analogous to pair potentials in mechanics.

**State-based** (implemented as `StatePeriDynamicAttention`):
```
state_i = aggregate({h_j - h_i for all j in horizon(i)})
state_j = aggregate({h_k - h_j for all k in horizon(j)})
score(i, j) = f_theta(state_i, state_j, h_j - h_i, pos)
```
Each token first summarizes its neighborhood's "deformation pattern," then interactions depend on both tokens' contexts. Strictly more expressive — can represent constitutive laws that bond-based cannot.

### The Mumford-Shah Connection

The strain energy + damage formulation implicitly minimizes a functional analogous to the **Mumford-Shah functional** (Mumford & Shah, 1989) from image segmentation:

```
E(u, K) = integral |grad u|^2 dx  +  alpha * length(K)  +  beta * integral (u - f)^2 dx
           strain energy             total damage            cross-entropy loss
           "be smooth"               "few cracks"            "predict well"
```

This jointly learns:
- **Where boundaries are** (damage = the discontinuity set K)
- **How to represent content within segments** (strain energy = smoothness prior)
- **How to predict the next token** (cross-entropy = task loss)

Standard attention has no smoothness prior and no concept of discontinuities.

### Three-Level Architecture

Peridynamics operates at three scales simultaneously:

```
TOKEN LEVEL (horizontal, sequence dimension):
  PeriDynamicAttention / StatePeriDynamicAttention
  - Bond-based or state-based interactions within causal horizon delta
  - Damage = bond breaking at semantic boundaries
  - Complexity: O(T * delta) per head

DEPTH LEVEL (vertical, layer dimension):
  Block Attention Residual
  - Layers grouped into blocks; each layer attends over all block summaries
  - State-based peridynamics over depth (learned weighted skip connections)
  - Optional damage gating to suppress cross-domain block summaries

PARALLELISM (speed, inference):
  DEER with damage-adaptive damping
  - Newton's method parallelizes sequential layer computation
  - Peridynamic damage provides per-layer predictability signal
  - Multigrid DEER uses block structure as coarse-grid checkpoints
```

---

## Architecture

### PeriDynamicAttention (bond-based)

```python
# For each token i, within causal horizon delta:
strain = disp[j] - disp[i]                    # relative deformation
bond_score = f_theta(strain, rel_pos)          # learned constitutive law
damage = sigmoid(g_theta(strain))              # bond breaking threshold
score = bond_score - damage * 10               # suppress broken bonds
output = softmax(score) @ values               # peridynamic integral
```

Key design choices:
- **Reduced bond dimension**: strains computed in `bd = head_size // bond_dim_ratio` space (configurable)
- **Damage initialization**: `sigmoid(-3) ~ 0.047` so bonds start mostly intact
- **Causal horizon**: left-padded sliding window, masked for positions < 0

### StatePeriDynamicAttention (state-based)

Adds neighborhood aggregation before bond scoring:

```python
# State aggregation: each token summarizes its neighborhood
strains = disp_window - disp_i                 # all bonds in horizon
state_i = mean(project(strains), over=horizon) # deformation state

# Bond score depends on both states + strain + position
score = f(strain_proj(strain) + state_i_proj(state_i)
        + state_j_proj(state_j) + pos_proj(rel_pos))
```

### Block Attention Residual (depth dimension)

```python
# Instead of x = x + layer(x), each layer sees all previous blocks:
h = softmax(proj * RMSNorm(blocks + [partial])) @ (blocks + [partial])
# With optional damage gating:
block_strain = norm(block_k) - norm(partial)
block_damage = sigmoid(damage_net(block_strain))
logits = logits - block_damage * 10  # suppress cross-domain blocks
```

### DEER Layer-Parallel Inference

Parallelizes `x^(l) = block_l(x^(l-1))` via Newton's method:
1. Guess all layer outputs simultaneously (Picard warm-start)
2. Linearize: diagonal Jacobian via finite differences or autograd
3. Solve linear recurrence via parallel associative scan: O(log L)
4. **Damage-adaptive damping**: `k_l = alpha * mean_damage(layer_l)`

---

## Repository Structure

```
perigpt/
  model.py                  # GPT + all attention variants + GPTConfig
  block_attn_res.py         # Block attention residual + damage gating + MultigridDEER
  deer_parallel.py          # DEER parallel forward + associative scan
  train.py                  # Training loop (nanoGPT-based, CUDA/MPS/CPU)
  sample.py                 # Text generation from checkpoints
  configurator.py           # Config file + CLI override system
  analyze_damage.py         # Damage vs domain boundary correlation analysis
  nanoperigpt_colab.ipynb   # Colab notebook with all experiments
  config/
    baseline.py             # Vanilla nanoGPT (standard attention)
    mixed_baseline.py       # Standard attention on mixed-domain data
  data/
    shakespeare_char/       # Character-level Shakespeare (~1.1M chars)
      prepare.py
    mixed_domain/           # 4-domain interleaved dataset (~4.4M chars)
      prepare.py            # Shakespeare + Python + JSON + Federalist Papers
```

## Quick Start

```bash
# Prepare data
python data/shakespeare_char/prepare.py
python data/mixed_domain/prepare.py

# Train baseline
python train.py config/baseline.py

# Train peridynamic attention
python train.py config/mixed_baseline.py \
    --attention_type=peridynamic --horizon=128 \
    --out_dir=out-peri --experiment_name=peri_h128

# Train state-based peridynamic attention
python train.py config/mixed_baseline.py \
    --attention_type=state_peridynamic --horizon=128 \
    --out_dir=out-state --experiment_name=state_h128

# Train with damage-gated block attention residual
python train.py config/mixed_baseline.py \
    --residual_type=block_attn --depth_block_size=3 --block_damage=True \
    --out_dir=out-blockdmg --experiment_name=blockres_damage

# Analyze damage patterns
python analyze_damage.py --out_dir=out-peri --n_samples=100

# Generate text
python sample.py --out_dir=out-peri
```

### Configuration

All peridynamic features are controlled via `GPTConfig` fields, overridable through config files or `--key=value` CLI arguments:

| Parameter | Default | Description |
|---|---|---|
| `attention_type` | `'standard'` | `'standard'`, `'peridynamic'`, `'state_peridynamic'`, `'hybrid'` |
| `horizon` | `32` | Peridynamic neighborhood size (causal window) |
| `bond_dim_ratio` | `4` | Bond dimension = head_size // ratio (lower = more capacity) |
| `residual_type` | `'standard'` | `'standard'` or `'block_attn'` |
| `depth_block_size` | `4` | Layers per block for block attention residual |
| `block_damage` | `False` | Enable damage gating in block attention |
| `energy_lambda` | `0.0` | Strain energy auxiliary loss weight |
| `deer_enabled` | `False` | Enable DEER parallel inference |
| `deer_method` | `'quasi-deer'` | `'deer'`, `'quasi-deer'`, `'elk'`, `'damage-elk'` |

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended; MPS/CPU supported)
- numpy, requests (for data preparation)

---

## Mixed-Domain Dataset

The critical testbed for peridynamic attention. Four domains interleaved in random chunks (64-384 characters):

| Domain | Source | Size | Style |
|---|---|---|---|
| Shakespeare | karpathy/char-rnn tinyshakespeare | 1.1M chars | Dramatic verse, archaic English |
| Python | pandas/core/frame.py | 650K chars | Code with indentation, docstrings, imports |
| JSON | mledoze/countries | 1.4M chars | Nested brackets, key-value pairs |
| Federalist Papers | Project Gutenberg | 1.2M chars | Formal 18th-century political prose |

Chunks are sized so domain boundaries frequently fall within the 256-character context window. No explicit separator tokens — the model must detect boundaries from content alone. Ground-truth domain labels are saved for damage analysis.

---

## Experimental Methodology

Following the [autoresearch](https://github.com/karpathy/autoresearch) methodology:
- **Single metric**: validation loss (cross-entropy)
- **Fixed budget**: 5000 iterations per experiment
- **One change at a time**: each experiment modifies exactly one variable
- **Append-only logging**: `results.tsv` tracks all experiments
- **Git-based tracking**: one commit per experiment

Model: 6 layers, 6 heads, 384 embedding dim, block_size=256, batch_size=64.
Training: AdamW, cosine LR schedule, 1e-3 peak LR, 0.2 dropout.

---

## Related Work

### Nonlocal Operators and Attention

- **Non-Local Neural Networks** (Wang et al., CVPR 2018) — Identified self-attention as a nonlocal operation, inspired by non-local means in image processing.
- **Nonlocal Attention Operator** (Yu, Silling et al., NeurIPS 2024) — Showed attention is equivalent to a double integral operator. Stewart Silling (inventor of peridynamics) is a coauthor. Works in the analysis direction (understanding attention via physics); PeriGPT works in the design direction (building attention from physics).
- **Continuum Attention for Neural Operators** (Calvello et al., 2024) — Reformulates attention as a map between infinite-dimensional function spaces. Applied to PDE solving, not sequence modeling.
- **Understanding Transformers through Continuous Dynamics** (Zhang & Zhou, 2024) — Models transformer depth as PDE evolution with attention as a nonlocal integral operator.

### Peridynamic Neural Networks

- **Peridynamic Neural Operators** (Jafarzadeh, Silling et al., 2024) — Learns nonlocal constitutive laws via neural operators grounded in state-based peridynamics. Applied to material simulation, not language.
- **MPNN for Bond-Associated Peridynamics** (Hu et al., 2024) — Message-passing networks as surrogates for peridynamic simulations.

### Depth-Wise Learned Skip Connections

- **Attention Residuals** (Kimi/Moonshot AI, March 2026) — Concurrent work. Replaces fixed residual connections with softmax attention over block summaries. Our work adds damage gating for domain-aware suppression, peridynamic interpretation, and DEER integration.
- **DenseFormer** (Pagliardini et al., NeurIPS 2024) — Learned scalar weights (not attention) for depth-wise aggregation.
- **Value Residual Learning / ResFormer** (ACL 2025) — Residual connections from first-layer values to all subsequent layers.

### Parallel Depth Computation

- **DEER** (Lim et al., ICLR 2024) — Newton's method + parallel scan for nonlinear sequential models.
- **ELK / quasi-DEER** (Gonzalez et al., NeurIPS 2024) — Stabilized DEER with damping.
- **Predictability Enables Parallelization** (Gonzalez et al., NeurIPS 2025) — Proves DEER convergence depends on system predictability. Provides theoretical justification for our damage-adaptive damping.

### Emergent Structure in Neural Networks

- **DINO** (Caron et al., ICCV 2021) — Self-supervised vision transformers learn object segmentation without labels. Closest precedent for emergent segmentation, but in vision.
- **Emergent Linguistic Structure** (Manning et al., PNAS 2020) — BERT learns syntactic structure (parse trees, coreference) without supervision. Demonstrates emergent structure at the syntactic level; we demonstrate it at the domain level via a novel mechanism.

### Variational Segmentation

- **Mumford-Shah Loss for Image Segmentation** (Kim & Ye, IEEE TIP 2020) — Mumford-Shah functional as a deep learning loss for 2D images. No prior application to 1D sequences or NLP.

---

## Citation

```bibtex
@misc{perigpt2026,
  title={PeriGPT: Peridynamic Attention for Heterogeneous Sequence Modeling},
  author={},
  year={2026},
  url={https://github.com/gokayfem/perigpt}
}
```

---

## License

MIT
