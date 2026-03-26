# PeriGPT: Peridynamic Attention for Heterogeneous Sequence Modeling

**Attention mechanisms derived from nonlocal continuum mechanics that discover structural boundaries through damage mechanics.**

PeriGPT maps peridynamic theory — a nonlocal reformulation of continuum mechanics designed for fracture and discontinuities — onto transformer attention. On heterogeneous multi-domain sequences, peridynamic attention outperforms both standard full attention and sliding window attention (Longformer/Mistral), with damage patterns that correlate with domain boundaries without any explicit supervision.

---

## Main Result

Peridynamic attention consistently outperforms standard attention **and** sliding window attention on heterogeneous data, while all depth-aggregation methods (ours, Kimi AttnRes, DenseFormer) hurt:

**Mixed-domain results** (Shakespeare + Python + JSON + Federalist Papers, 4.4M chars):

| Method | Source | val_loss | vs baseline |
|---|---|---|---|
| **Peri bd=32 h=128** | **Ours** | **0.8414** | **-0.0170** |
| Peri h=128 | Ours | 0.8489 | -0.0095 |
| Peri h=64 | Ours | 0.8515 | -0.0069 |
| Peri h=32 | Ours | 0.8542 | -0.0042 |
| Hybrid (peri + global) | Ours | 0.8558 | -0.0026 |
| Sliding window h=128 | Longformer/Mistral | 0.8572 | -0.0012 |
| Standard attention | nanoGPT | 0.8584 | — |
| Sliding window h=32 | Longformer/Mistral | 0.8670 | +0.0086 |
| Kimi AttnRes | arXiv:2603.15031 | 0.8800 | +0.0216 |
| Block AttnRes | Ours | 0.8807 | +0.0223 |
| DenseFormer DWA | NeurIPS 2024 | 1.1000 | +0.2416 |

**Shakespeare results** (homogeneous text, 1.1M chars):

| Method | Source | val_loss | vs baseline |
|---|---|---|---|
| DenseFormer DWA | NeurIPS 2024 | 1.5511 | -0.1496 |
| Kimi AttnRes | arXiv:2603.15031 | 1.5958 | -0.1049 |
| Block AttnRes | Ours | 1.6085 | -0.0922 |
| Standard attention | nanoGPT | 1.7007 | — |
| Sliding window h=128 | Longformer/Mistral | 1.7014 | +0.0007 |
| Peri h=128 | Ours | 1.7992 | +0.0985 |
| Sliding window h=32 | Longformer/Mistral | 1.8181 | +0.1174 |
| Peri h=32 | Ours | 1.8136 | +0.1129 |

The ranking **inverts** between homogeneous and heterogeneous data — exactly as the peridynamic theory predicts.

---

## Key Ablation: Peridynamic vs Sliding Window

Same bounded horizon, same O(T * delta) complexity. The only difference: strain-based scoring + damage (peridynamic) vs standard Q*K dot product (sliding window). This isolates the contribution of the peridynamic mechanism.

| Dataset | Horizon | Peri | SWA | Delta | Winner |
|---|---|---|---|---|---|
| Mixed | h=32 | 0.8542 | 0.8670 | **-0.0128** | **Peri** |
| Mixed | h=128 | 0.8489 | 0.8572 | **-0.0083** | **Peri** |
| Shakespeare | h=32 | 1.8136 | 1.8181 | -0.0045 | Peri (barely) |
| Shakespeare | h=128 | 1.7992 | 1.7014 | +0.0978 | SWA |

On heterogeneous data, strain + damage wins at every horizon. On homogeneous data with large windows, standard Q*K wins (flash attention is highly optimized for this regime).

---

## Damage Learns Domain Boundaries

Without any labels or explicit boundary signal, the peridynamic damage mechanism learns to assign higher damage at domain transitions:

```
Mean damage at domain BOUNDARIES:  0.4204 (+-0.1593)
Mean damage at domain INTERIOR:    0.3594 (+-0.0068)
Ratio (boundary / interior):       1.17x
```

All 6 transformer layers detect boundaries. Interior damage has extremely low variance (+-0.0068), indicating stable within-domain behavior. Boundary variance is high (+-0.1593), reflecting that some transitions are sharper than others (Shakespeare-to-JSON is obvious; Shakespeare-to-Federalist-Papers is subtle — both are formal English).

This is **emergent structural segmentation** — the model discovers domain boundaries as a byproduct of next-token prediction, mediated by the physics of the damage mechanism.

---

## Negative Results (equally important)

### Depth-aggregation methods hurt on heterogeneous data

All three depth-aggregation approaches — our Block AttnRes, Kimi AttnRes, and DenseFormer — hurt on mixed-domain data:

| Method | Shakespeare | Mixed |
|---|---|---|
| DenseFormer DWA | **1.5511** (best) | 1.1000 (catastrophic) |
| Kimi AttnRes | 1.5958 | 0.8800 (+0.0216) |
| Block AttnRes (ours) | 1.6085 | 0.8807 (+0.0223) |
| Baseline | 1.7007 | 0.8584 |

DenseFormer's extreme behavior is informative: static scalar weights provide strong regularization on homogeneous data but completely fail on domain shifts. Content-dependent routing (Kimi, ours) degrades more gracefully but still hurts. Cross-layer skip connections route information from previous blocks that may belong to a different domain, contaminating the current computation.

### Damage gating in depth dimension does not help

Our Block AttnRes with damage gating (0.8783) slightly outperforms our Block AttnRes without it (0.8807) and Kimi's version (0.8800) on mixed data, but all three still lose to the baseline. The damage mechanism partially mitigates cross-domain contamination but not enough to overcome the fundamental issue.

### Full stack does not compound

Combining peridynamic attention (sequence) + block attn-res (depth) yields 0.8610 — worse than pure peridynamic attention (0.8489). The depth-dimension contamination outweighs whatever the block structure provides.

---

## Theoretical Framework

### The Peridynamic-Attention Isomorphism

Peridynamics (Silling, 2000) replaces local PDEs with a nonlocal integral equation where material points interact through bond forces that depend on **relative deformation** (strain). When strain exceeds a critical threshold, bonds **break** — this is how cracks propagate.

| Peridynamics | PeriGPT |
|---|---|
| Material point | Token embedding |
| Horizon delta | Causal sliding window |
| Bond strain (x' - x) | Relative feature difference (h_j - h_i) |
| Micro-modulus / force function | Learned bond score function |
| Bond damage | Adaptive sparsity via learned critical-strain threshold |
| Deformation state | Neighborhood aggregation before scoring |
| Strain energy density | Auxiliary regularization loss |

Key distinction from standard attention: peridynamic attention operates on **relative differences** (strain) between tokens, not absolute similarity (Q*K). And the damage mechanism provides **content-dependent adaptive sparsity** — bonds that cross semantic discontinuities are suppressed.

### The Mumford-Shah Connection

The strain energy + damage formulation implicitly minimizes a functional analogous to the Mumford-Shah functional from image segmentation:

```
E(u, K) = integral |grad u|^2 dx  +  alpha * length(K)  +  beta * integral (u - f)^2 dx
           strain energy             total damage            cross-entropy loss
           "be smooth"               "few cracks"            "predict well"
```

This jointly learns where boundaries are (damage), how to represent content within segments (strain energy), and how to predict the next token (cross-entropy). No prior work has connected the Mumford-Shah functional to sequence modeling.

---

## Architecture

### PeriDynamicAttention

```python
# For each token i, within causal horizon delta:
strain = disp[j] - disp[i]                    # relative deformation
bond_score = f_theta(strain, rel_pos)          # learned constitutive law
damage = sigmoid(g_theta(strain))              # bond breaking threshold
score = bond_score - damage * 10               # suppress broken bonds
output = softmax(score) @ values               # peridynamic integral
```

### StatePeriDynamicAttention

Adds neighborhood aggregation — each token summarizes its horizon's deformation pattern before computing bond scores:

```python
state_i = mean(project(strains_in_horizon_i))  # "what my neighborhood looks like"
state_j = mean(project(strains_in_horizon_j))  # "what your neighborhood looks like"
score = f(strain_ij, state_i, state_j, pos)     # interaction depends on both contexts
```

### Comparison Baselines Implemented

| Method | File | Description |
|---|---|---|
| `SlidingWindowAttention` | model.py | Longformer/Mistral-style: same horizon, standard Q*K, no damage |
| `KimiAttnResLayer` | block_attn_res.py | arXiv:2603.15031: block attention without damage gating |
| `DenseFormerLayer` | block_attn_res.py | NeurIPS 2024: learned scalar weights over depth |

---

## Repository Structure

```
perigpt/
  model.py                  # All attention variants + GPT
  block_attn_res.py         # Block AttnRes + Kimi + DenseFormer baselines
  deer_parallel.py          # DEER parallel forward + associative scan
  train.py                  # Training loop (CUDA/MPS/CPU)
  sample.py                 # Text generation
  analyze_damage.py         # Damage vs domain boundary analysis
  configurator.py           # Config system
  nanoperigpt_colab.ipynb   # All experiments
  config/
    baseline.py             # Standard attention, shakespeare
    mixed_baseline.py       # Standard attention, mixed-domain
  data/
    shakespeare_char/       # 1.1M chars, vocab 65
    mixed_domain/           # 4.4M chars, vocab 111, 4 domains interleaved
```

## Quick Start

```bash
# Prepare data
python data/shakespeare_char/prepare.py
python data/mixed_domain/prepare.py

# Baseline
python train.py config/mixed_baseline.py

# Peridynamic attention (best config)
python train.py config/mixed_baseline.py \
    --attention_type=peridynamic --horizon=128 --bond_dim_ratio=2 \
    --out_dir=out-peri --experiment_name=peri_best

# Sliding window baseline (same horizon, no damage)
python train.py config/mixed_baseline.py \
    --attention_type=sliding_window --horizon=128 \
    --out_dir=out-swa --experiment_name=swa_h128

# Analyze damage patterns
python analyze_damage.py --out_dir=out-peri

# Generate text
python sample.py --out_dir=out-peri
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `attention_type` | `standard` | `standard`, `peridynamic`, `state_peridynamic`, `hybrid`, `sliding_window` |
| `horizon` | `32` | Neighborhood size (for peridynamic and sliding_window) |
| `bond_dim_ratio` | `4` | Bond dim = head_size // ratio (2 recommended) |
| `residual_type` | `standard` | `standard`, `block_attn`, `kimi_attn_res`, `denseformer` |
| `depth_block_size` | `4` | Layers per block for block attention |
| `block_damage` | `False` | Damage gating in block attention |

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- numpy, requests

---

## Mixed-Domain Dataset

Four domains interleaved in random chunks (64-384 chars) so domain boundaries fall within the 256-char context window:

| Domain | Source | Size | Style |
|---|---|---|---|
| Shakespeare | tinyshakespeare | 1.1M chars | Dramatic verse, archaic English |
| Python | pandas/core/frame.py | 650K chars | Code with indentation, docstrings |
| JSON | mledoze/countries | 1.4M chars | Nested brackets, key-value pairs |
| Federalist Papers | Project Gutenberg | 1.2M chars | Formal 18th-century political prose |

No explicit separator tokens. Ground-truth domain labels saved for damage analysis.

---

## Related Work

**Nonlocal operators and attention.** Wang et al. (CVPR 2018) identified self-attention as a nonlocal operation. Yu, Silling et al. (NeurIPS 2024) showed attention is equivalent to a double integral operator — with Stewart Silling (inventor of peridynamics) as coauthor. These work in the analysis direction; we work in the design direction.

**Peridynamic neural operators.** Jafarzadeh, Silling et al. (2024) learn nonlocal constitutive laws for material simulation. We apply the same mathematical structure to language.

**Efficient attention.** Sliding window attention (Longformer, Beltagy et al. 2020; Mistral, Jiang et al. 2023) uses a bounded window with standard Q*K scoring. We show that replacing Q*K with strain-based scoring + damage improves results on heterogeneous data at the same complexity.

**Depth-wise aggregation.** DenseFormer (Pagliardini et al., NeurIPS 2024) uses learned scalar weights. Attention Residuals (Kimi/Moonshot AI, March 2026) uses learned attention over block summaries. We compare against both and find they help on homogeneous data but hurt on heterogeneous data.

**Emergent structure.** DINO (Caron et al., ICCV 2021) showed self-supervised vision transformers learn object segmentation. Manning et al. (PNAS 2020) showed BERT learns syntax. We show peridynamic damage learns domain segmentation — a different level of structure via a novel mechanism.

**Variational segmentation.** The Mumford-Shah functional (1989) jointly optimizes smoothness and discontinuity placement for images. Kim & Ye (IEEE TIP 2020) applied it with deep learning for 2D images. Our strain energy + damage formulation is the first connection to 1D sequence modeling.

**DEER.** Lim et al. (ICLR 2024) parallelize sequential models via Newton's method. Gonzalez et al. (NeurIPS 2024, 2025) stabilize with damping and prove convergence depends on predictability. Our damage-adaptive damping uses peridynamic damage as a predictability proxy.

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

## License

MIT
