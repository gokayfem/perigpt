"""
Prepare SlimPajama for GPT-2 pretraining.

Downloads DKYoon/SlimPajama-6B (curated 6B token subset of
cerebras/SlimPajama-627B), tokenizes with GPT-2 BPE via tiktoken,
and saves as train.bin/val.bin in nanoGPT format.

SlimPajama contains 7 domains: CommonCrawl, C4, GitHub, Books,
ArXiv, Wikipedia, StackExchange — naturally heterogeneous.
Domain labels are saved for damage analysis.

Usage:
    pip install datasets tiktoken
    python data/slimpajama/prepare.py
    python data/slimpajama/prepare.py --num_tokens 100000000  # 100M for quick test
"""
import os
import pickle
import argparse
import numpy as np

data_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tokens', type=int, default=500_000_000,
                        help='Target token count (default: 500M)')
    parser.add_argument('--val_fraction', type=float, default=0.005,
                        help='Validation fraction (default: 0.5%%)')
    args = parser.parse_args()

    print("=== SlimPajama Data Preparation ===")
    print(f"Target: {args.num_tokens / 1e6:.0f}M tokens\n")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = 50304  # GPT-2 50257 rounded to nearest 64
    print(f"Tokenizer: GPT-2 BPE (vocab={enc.n_vocab}, using {vocab_size})")

    from datasets import load_dataset
    print("Loading DKYoon/SlimPajama-6B (streaming)...")
    ds = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)

    all_tokens = []
    domain_boundaries = []  # (position, domain_name) at each document start
    domain_counts = {}
    n_docs = 0
    current_pos = 0

    for doc in ds:
        text = doc["text"]
        meta = doc.get("meta", {})
        if isinstance(meta, dict):
            domain = meta.get("redpajama_set_name", "unknown")
        else:
            domain = "unknown"

        tokens = enc.encode_ordinary(text)
        if not tokens:
            continue

        domain_boundaries.append((current_pos, domain))
        all_tokens.extend(tokens)
        current_pos += len(tokens)
        domain_counts[domain] = domain_counts.get(domain, 0) + len(tokens)
        n_docs += 1

        if n_docs % 10000 == 0:
            print(f"  {n_docs:>8,} docs  {len(all_tokens)/1e6:>8.1f}M tokens")

        if len(all_tokens) >= args.num_tokens:
            break

    n = min(len(all_tokens), args.num_tokens)
    all_tokens = all_tokens[:n]

    print(f"\nProcessed: {n:,} tokens from {n_docs:,} documents")
    print(f"\nDomain distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain:30s} {count/1e6:>8.1f}M  ({100*count/n:.1f}%)")

    # Split
    val_size = int(n * args.val_fraction)
    train_size = n - val_size
    train_tokens = np.array(all_tokens[:train_size], dtype=np.uint16)
    val_tokens = np.array(all_tokens[train_size:], dtype=np.uint16)
    print(f"\nSplit: {train_size/1e6:.1f}M train, {val_size/1e6:.1f}M val")

    # Save tokens
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    print(f"Saved: train.bin ({os.path.getsize(train_path)/1e9:.1f}GB)")
    print(f"Saved: val.bin ({os.path.getsize(val_path)/1e6:.0f}MB)")

    # Save meta (for nanoGPT compatibility)
    meta = {
        'vocab_size': vocab_size,
        'tokenizer': 'gpt2',
        'num_train_tokens': train_size,
        'num_val_tokens': val_size,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # Save domain boundaries for damage analysis
    # Each entry: (token_position, domain_name)
    labels = {
        'domain_boundaries': domain_boundaries,
        'domains': sorted(domain_counts.keys()),
        'domain_counts': domain_counts,
        'total_tokens': n,
        'train_size': train_size,
    }
    with open(os.path.join(data_dir, 'domain_labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)
    print(f"Saved: domain_labels.pkl ({len(domain_boundaries)} boundaries)")

    print("\nDone!")


if __name__ == "__main__":
    main()
