"""
Prepare a mixed-domain character-level dataset for testing peridynamic damage.

Interleaves chunks from 4 very different domains:
  1. Shakespeare (dramatic verse, archaic English)
  2. Python source code (pandas DataFrame implementation)
  3. JSON (structured country data)
  4. Federalist Papers (formal 18th-century political prose)

Chunks are randomly sized (64-384 chars) so that domain boundaries
frequently fall within a block_size=256 training window. This is the
key test: can peridynamic damage learn to break bonds at domain
boundaries without any explicit separator?

No special separator tokens — domains are concatenated directly with
just a double newline. The model must learn to detect boundaries from
content alone.
"""
import os
import pickle
import random
import requests
import numpy as np

random.seed(42)

SOURCES = {
    'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    'python': 'https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/core/frame.py',
    'json': 'https://raw.githubusercontent.com/mledoze/countries/master/countries.json',
    'federalist': 'https://www.gutenberg.org/cache/epub/1404/pg1404.txt',
}

MIN_CHUNK = 64
MAX_CHUNK = 384

data_dir = os.path.dirname(__file__)


def download_sources():
    """Download all source texts."""
    texts = {}
    for name, url in SOURCES.items():
        cache_path = os.path.join(data_dir, f'{name}_raw.txt')
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8', errors='replace') as f:
                texts[name] = f.read()
            print(f"  {name}: {len(texts[name]):,} chars (cached)")
        else:
            print(f"  downloading {name}...")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            text = resp.text
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
            texts[name] = text
            print(f"  {name}: {len(text):,} chars")
    return texts


def chunkify(text, min_size, max_size):
    """Split text into random-sized chunks."""
    chunks = []
    i = 0
    while i < len(text):
        size = random.randint(min_size, max_size)
        chunk = text[i:i + size]
        if len(chunk) >= min_size // 2:
            chunks.append(chunk)
        i += size
    return chunks


def interleave_chunks(domain_chunks):
    """
    Interleave chunks from different domains randomly.
    Each chunk is from a single domain, so domain boundaries
    create natural discontinuities in the stream.
    """
    all_chunks = []
    for domain_name, chunks in domain_chunks.items():
        for chunk in chunks:
            all_chunks.append((domain_name, chunk))

    random.shuffle(all_chunks)
    return all_chunks


def build_dataset(texts):
    """Build the interleaved mixed-domain text."""
    print("\nChunkifying domains...")
    domain_chunks = {}
    for name, text in texts.items():
        chunks = chunkify(text, MIN_CHUNK, MAX_CHUNK)
        domain_chunks[name] = chunks
        print(f"  {name}: {len(chunks)} chunks")

    print("\nInterleaving...")
    interleaved = interleave_chunks(domain_chunks)

    # Build the final text with double-newline separators
    # Also build a domain label array for analysis
    final_text = ""
    domain_labels = []  # parallel array: domain name per character

    for domain_name, chunk in interleaved:
        start = len(final_text)
        final_text += chunk + "\n\n"
        domain_labels.extend([domain_name] * (len(chunk) + 2))

    print(f"\nTotal interleaved text: {len(final_text):,} chars")

    # Domain distribution
    from collections import Counter
    label_counts = Counter(domain_labels)
    for name, count in sorted(label_counts.items()):
        pct = 100 * count / len(domain_labels)
        print(f"  {name}: {count:,} chars ({pct:.1f}%)")

    return final_text, domain_labels


def main():
    print("=== Mixed-Domain Dataset Preparation ===\n")
    print("Downloading sources...")
    texts = download_sources()

    final_text, domain_labels = build_dataset(texts)

    # Character-level encoding (same as shakespeare_char)
    chars = sorted(list(set(final_text)))
    vocab_size = len(chars)
    print(f"\nVocab size: {vocab_size}")
    print(f"Characters: {''.join(chars[:50])}...")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    # Train/val split
    n = len(final_text)
    train_text = final_text[:int(n * 0.9)]
    val_text = final_text[int(n * 0.9):]
    train_labels = domain_labels[:int(n * 0.9)]
    val_labels = domain_labels[int(n * 0.9):]

    train_ids = encode(train_text)
    val_ids = encode(val_text)
    print(f"Train: {len(train_ids):,} tokens")
    print(f"Val: {len(val_ids):,} tokens")

    # Save binary files
    np.array(train_ids, dtype=np.uint16).tofile(
        os.path.join(data_dir, 'train.bin'))
    np.array(val_ids, dtype=np.uint16).tofile(
        os.path.join(data_dir, 'val.bin'))

    # Save meta
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # Save domain labels for analysis (which domain each character belongs to)
    # This lets us analyze whether damage correlates with domain boundaries
    labels_meta = {
        'train_labels': train_labels,
        'val_labels': val_labels,
        'domains': list(SOURCES.keys()),
    }
    with open(os.path.join(data_dir, 'domain_labels.pkl'), 'wb') as f:
        pickle.dump(labels_meta, f)

    print(f"\nSaved: train.bin, val.bin, meta.pkl, domain_labels.pkl")
    print("Done!")


if __name__ == '__main__':
    main()
