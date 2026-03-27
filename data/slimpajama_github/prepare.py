"""
Prepare GitHub-only subset of SlimPajama.

Code is naturally heterogeneous — a single file contains prose (docstrings),
code syntax, SQL strings, regex, JSON configs, URLs, type annotations.
This is the ideal test case for peridynamic damage without any synthetic mixing.

Usage:
    pip install datasets tiktoken
    python data/slimpajama_github/prepare.py
    python data/slimpajama_github/prepare.py --num_tokens 100000000  # 100M quick
"""
import os
import pickle
import argparse
import numpy as np

data_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tokens', type=int, default=200_000_000,
                        help='Target tokens (default: 200M)')
    parser.add_argument('--val_fraction', type=float, default=0.005)
    args = parser.parse_args()

    print("=== SlimPajama GitHub-Only Data Preparation ===")
    print(f"Target: {args.num_tokens / 1e6:.0f}M tokens\n")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = 50304

    from datasets import load_dataset
    print("Loading DKYoon/SlimPajama-6B (streaming, filtering GitHub)...")
    ds = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)

    all_tokens = []
    n_docs = 0
    n_skipped = 0

    for doc in ds:
        meta = doc.get("meta", {})
        if isinstance(meta, dict):
            domain = meta.get("redpajama_set_name", "")
        else:
            domain = ""

        if "Github" not in domain and "github" not in domain.lower():
            n_skipped += 1
            if n_skipped % 100000 == 0:
                print(f"  skipped {n_skipped:,} non-GitHub docs...")
            continue

        text = doc["text"]
        tokens = enc.encode_ordinary(text)
        if not tokens:
            continue

        all_tokens.extend(tokens)
        n_docs += 1

        if n_docs % 5000 == 0:
            print(f"  {n_docs:>8,} GitHub docs  {len(all_tokens)/1e6:>8.1f}M tokens")

        if len(all_tokens) >= args.num_tokens:
            break

    n = min(len(all_tokens), args.num_tokens)
    all_tokens = all_tokens[:n]
    print(f"\nTotal: {n:,} tokens from {n_docs:,} GitHub documents")
    print(f"(skipped {n_skipped:,} non-GitHub documents)")

    # Split
    val_size = int(n * args.val_fraction)
    train_size = n - val_size
    train_tokens = np.array(all_tokens[:train_size], dtype=np.uint16)
    val_tokens = np.array(all_tokens[train_size:], dtype=np.uint16)
    print(f"Split: {train_size/1e6:.1f}M train, {val_size/1e6:.1f}M val")

    # Save
    train_tokens.tofile(os.path.join(data_dir, "train.bin"))
    val_tokens.tofile(os.path.join(data_dir, "val.bin"))

    meta = {'vocab_size': vocab_size, 'tokenizer': 'gpt2',
            'num_train_tokens': train_size, 'num_val_tokens': val_size}
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved: train.bin ({os.path.getsize(os.path.join(data_dir, 'train.bin'))/1e6:.0f}MB)")
    print("Done!")


if __name__ == "__main__":
    main()
