"""
Run GitHub code experiments: standard vs peridynamic vs sliding window.

Code is the most naturally heterogeneous domain — docstrings, syntax,
strings, regex, configs, all in one file. No synthetic mixing needed.

Usage:
    python run_github_experiments.py
    python run_github_experiments.py --seeds 1 --max_iters 1000  # quick
"""
import os
import sys
import subprocess
import argparse


def run(name, overrides, seed, base_config='config/github_baseline.py', max_iters=None):
    cmd = [sys.executable, 'train.py', base_config]
    cmd.append(f'--out_dir=out-gh-{name}-s{seed}')
    cmd.append(f'--experiment_name=gh_{name}_s{seed}')
    cmd.append(f'--wandb_run_name=gh_{name}_s{seed}')
    for k, v in overrides.items():
        cmd.append(f'--{k}={v}')
    if max_iters:
        cmd.append(f'--max_iters={max_iters}')

    print(f"\n{'='*60}")
    print(f"  gh_{name}_s{seed}")
    print(f"{'='*60}\n")

    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--max_iters', type=int, default=None)
    args = parser.parse_args()

    experiments = {
        'baseline': {'attention_type': 'standard'},
        'peri_h64': {'attention_type': 'peridynamic', 'horizon': 64, 'bond_dim_ratio': 2},
        'peri_h128': {'attention_type': 'peridynamic', 'horizon': 128, 'bond_dim_ratio': 2},
        'swa_h64': {'attention_type': 'sliding_window', 'horizon': 64},
        'swa_h128': {'attention_type': 'sliding_window', 'horizon': 128},
    }

    n_total = len(experiments) * args.seeds
    print(f"GitHub Code Experiments: {len(experiments)} configs × {args.seeds} seeds = {n_total} runs")

    if not os.path.exists('data/slimpajama_github/train.bin'):
        print("\nERROR: data/slimpajama_github/train.bin not found!")
        print("Run: python data/slimpajama_github/prepare.py")
        sys.exit(1)

    for name, overrides in experiments.items():
        for seed in range(args.seeds):
            run(name, overrides, seed, max_iters=args.max_iters)

    print(f"\nDone! Check results.tsv")


if __name__ == "__main__":
    main()
