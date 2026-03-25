"""
Sample from a trained nanoPeriGPT model.

Usage:
    python sample.py --out_dir=out-baseline
    python sample.py --out_dir=out-baseline --num_samples=5 --max_new_tokens=500
"""

import os
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out-baseline'
start = "\n"
num_samples = 3
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
dtype = 'float32'
compile = False
# -----------------------------------------------------------------------------

exec(open('configurator.py').read())

torch.manual_seed(seed)

ctx = nullcontext()

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

if compile:
    model = torch.compile(model)

# look for dataset meta for encoding
dataset = checkpoint['config'].get('dataset', 'shakespeare_char')
meta_path = os.path.join('data', dataset, 'meta.pkl')
if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    raise FileNotFoundError(f"No meta.pkl found at {meta_path}")

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---')
