"""
Deep performance analysis: benchmark every optimization on CUDA.
Tests dtype, TF32, linearity trick, fused projections, horizon reduction.

Usage: python bench_deep.py  (on CUDA)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "Needs CUDA"
device = 'cuda'


def bench(fn, label, n_warmup=10, n_runs=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f'  {label:55s} {ms:8.2f} ms')
    return ms


def bench_model(attn_type, label, horizon=128, bd_ratio=2, dtype=torch.float32,
                tf32=False, do_compile=False, B=64, T=256):
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    from model import GPTConfig, GPT
    cfg = GPTConfig(block_size=T, vocab_size=65, n_layer=6, n_head=6, n_embd=384,
                    dropout=0.0, bias=False, deer_enabled=False, energy_lambda=0.0,
                    attention_type=attn_type, horizon=horizon, bond_dim_ratio=bd_ratio)
    model = GPT(cfg).to(device).to(dtype).eval()
    if do_compile:
        model = torch.compile(model, mode='max-autotune')
    x = torch.randint(0, 65, (B, T), device=device)

    def run():
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            model(x)

    ms = bench(run, label)

    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.backends.cudnn.allow_tf32 = old_tf32
    del model
    torch.cuda.empty_cache()
    return ms


print("=" * 70)
print("DEEP PERFORMANCE ANALYSIS — RTX 6000 Pro Blackwell")
print("B=64, T=256, 6 layers, 6 heads, 384 embd")
print("=" * 70)

# === 1. DTYPE + TF32 IMPACT ===
print("\n--- 1. DTYPE AND TF32 IMPACT ---")
print("    (the warning said TF32 was disabled — this could be huge)\n")

t_std_f32 = bench_model('standard', 'Standard flash — float32', dtype=torch.float32)
t_std_f32_tf32 = bench_model('standard', 'Standard flash — float32 + TF32', dtype=torch.float32, tf32=True)
t_std_bf16 = bench_model('standard', 'Standard flash — bfloat16', dtype=torch.bfloat16)
t_std_bf16_c = bench_model('standard', 'Standard flash — bfloat16 + compile', dtype=torch.bfloat16, do_compile=True)

print()
t_peri_f32 = bench_model('peridynamic', 'Peri — float32', dtype=torch.float32)
t_peri_f32_tf32 = bench_model('peridynamic', 'Peri — float32 + TF32', dtype=torch.float32, tf32=True)
t_peri_bf16 = bench_model('peridynamic', 'Peri — bfloat16', dtype=torch.bfloat16)
t_peri_bf16_c = bench_model('peridynamic', 'Peri — bfloat16 + compile', dtype=torch.bfloat16, do_compile=True)
t_peri_f32_c = bench_model('peridynamic', 'Peri — float32 + TF32 + compile', dtype=torch.float32, tf32=True, do_compile=True)

print(f"\n  DTYPE IMPACT ON PERI:")
print(f"    float32:              {t_peri_f32:6.1f} ms  (1.00x)")
print(f"    float32 + TF32:       {t_peri_f32_tf32:6.1f} ms  ({t_peri_f32/t_peri_f32_tf32:.2f}x)")
print(f"    bfloat16:             {t_peri_bf16:6.1f} ms  ({t_peri_f32/t_peri_bf16:.2f}x)")
print(f"    bf16 + compile:       {t_peri_bf16_c:6.1f} ms  ({t_peri_f32/t_peri_bf16_c:.2f}x)")
print(f"    f32 + TF32 + compile: {t_peri_f32_c:6.1f} ms  ({t_peri_f32/t_peri_f32_c:.2f}x)")

# === 2. HORIZON IMPACT ===
print("\n--- 2. HORIZON SIZE IMPACT (bf16 + compile) ---\n")

for h in [32, 64, 128, 256]:
    t = bench_model('peridynamic', f'Peri bf16+compile h={h}',
                    horizon=h, dtype=torch.bfloat16, do_compile=True)

# === 3. BOND_DIM IMPACT ===
print("\n--- 3. BOND_DIM RATIO IMPACT (bf16 + compile, h=128) ---\n")

for r in [1, 2, 4, 8]:
    bd = max(64 // r, 8)
    t = bench_model('peridynamic', f'Peri bf16+compile bd_ratio={r} (bd={bd})',
                    bd_ratio=r, dtype=torch.bfloat16, do_compile=True)

# === 4. SLIDING WINDOW COMPARISON (same conditions) ===
print("\n--- 4. FAIR COMPARISON: same dtype, same compile ---\n")

t_std_best = bench_model('standard', 'Standard — bf16 + compile',
                          dtype=torch.bfloat16, do_compile=True)
t_swa_best = bench_model('sliding_window', 'SWA h=128 — bf16 + compile',
                          horizon=128, dtype=torch.bfloat16, do_compile=True)
t_swa_64 = bench_model('sliding_window', 'SWA h=64 — bf16 + compile',
                        horizon=64, dtype=torch.bfloat16, do_compile=True)
t_peri_best = bench_model('peridynamic', 'Peri h=128 — bf16 + compile',
                           horizon=128, dtype=torch.bfloat16, do_compile=True)
t_peri_64 = bench_model('peridynamic', 'Peri h=64 — bf16 + compile',
                         horizon=64, dtype=torch.bfloat16, do_compile=True)
t_peri_64_r4 = bench_model('peridynamic', 'Peri h=64 bd=16 — bf16 + compile',
                            horizon=64, bd_ratio=4, dtype=torch.bfloat16, do_compile=True)

print(f"\n{'='*70}")
print(f"FINAL COMPARISON (all bf16 + compile)")
print(f"{'='*70}")
print(f"  {'Method':<45s} {'ms':>8s} {'vs std':>8s}")
print(f"  {'-'*65}")
print(f"  {'Standard (flash)':<45s} {t_std_best:8.2f} {'1.00x':>8s}")
print(f"  {'SWA h=64':<45s} {t_swa_64:8.2f} {f'{t_swa_64/t_std_best:.2f}x':>8s}")
print(f"  {'SWA h=128':<45s} {t_swa_best:8.2f} {f'{t_swa_best/t_std_best:.2f}x':>8s}")
print(f"  {'Peri h=64 bd=16':<45s} {t_peri_64_r4:8.2f} {f'{t_peri_64_r4/t_std_best:.2f}x':>8s}")
print(f"  {'Peri h=64':<45s} {t_peri_64:8.2f} {f'{t_peri_64/t_std_best:.2f}x':>8s}")
print(f"  {'Peri h=128':<45s} {t_peri_best:8.2f} {f'{t_peri_best/t_std_best:.2f}x':>8s}")
print(f"\n  Previous (f32, no TF32, no compile):  {t_peri_f32/t_std_f32:.1f}x slower")
print(f"  Current best peri:                     {t_peri_64_r4/t_std_best:.1f}x slower")
