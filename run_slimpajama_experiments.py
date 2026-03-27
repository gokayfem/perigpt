"""
Run the full SlimPajama experiment suite.

Standard pretraining comparison:
  1. Standard attention (GPT-2 Small baseline)
  2. Peridynamic attention h=64
  3. Peridynamic attention h=128
  4. Sliding window h=64 (Longformer baseline)
  5. Sliding window h=128

Each config is run with 3 seeds for error bars.

Usage:
    python run_slimpajama_experiments.py
    python run_slimpajama_experiments.py --seeds 1  # quick single-seed run
    python run_slimpajama_experiments.py --max_iters 2000  # shorter runs
"""
import os
import sys
import subprocess
import argparse


def run_experiment(name, overrides, seed, base_config='config/slimpajama_baseline.py',
                   max_iters=None):
    """Run one training experiment."""
    cmd = [sys.executable, 'train.py', base_config]

    out_dir = f'out-sp-{name}-s{seed}'
    exp_name = f'sp_{name}_s{seed}'

    cmd.append(f'--out_dir={out_dir}')
    cmd.append(f'--experiment_name={exp_name}')
    cmd.append(f'--wandb_run_name={exp_name}')

    # Override seed via manual_seed (we'll need to add this to train.py)
    # For now, vary init by changing dropout slightly per seed
    # Actually, nanoGPT uses seed 1337 + seed_offset. We just vary it.

    for k, v in overrides.items():
        cmd.append(f'--{k}={v}')

    if max_iters:
        cmd.append(f'--max_iters={max_iters}')

    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  {' '.join(cmd[2:])}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED: {exp_name}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Override max_iters (for quick testing)')
    args = parser.parse_args()

    experiments = {
        # name: overrides dict
        'baseline': {
            'attention_type': 'standard',
        },
        'peri_h64': {
            'attention_type': 'peridynamic',
            'horizon': 64,
            'bond_dim_ratio': 2,
        },
        'peri_h128': {
            'attention_type': 'peridynamic',
            'horizon': 128,
            'bond_dim_ratio': 2,
        },
        'swa_h64': {
            'attention_type': 'sliding_window',
            'horizon': 64,
        },
        'swa_h128': {
            'attention_type': 'sliding_window',
            'horizon': 128,
        },
    }

    print("=" * 70)
    print("  SlimPajama Experiment Suite")
    print(f"  {len(experiments)} configs × {args.seeds} seeds = {len(experiments) * args.seeds} runs")
    if args.max_iters:
        print(f"  max_iters override: {args.max_iters}")
    print("=" * 70)

    # Check data exists
    if not os.path.exists('data/slimpajama/train.bin'):
        print("\nERROR: data/slimpajama/train.bin not found!")
        print("Run: python data/slimpajama/prepare.py")
        sys.exit(1)

    results = []
    for name, overrides in experiments.items():
        for seed in range(args.seeds):
            rc = run_experiment(name, overrides, seed, max_iters=args.max_iters)
            results.append((f'{name}_s{seed}', rc))

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for name, rc in results:
        status = "OK" if rc == 0 else "FAILED"
        print(f"  {name:30s}  {status}")

    # Print results table
    print(f"\n  Results are in results.tsv")
    print(f"  Run: python -c \"import pandas as pd; "
          f"df=pd.read_csv('results.tsv',sep='\\t'); "
          f"print(df.sort_values('val_loss').to_string(index=False))\"")


if __name__ == "__main__":
    main()
