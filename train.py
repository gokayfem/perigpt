"""
nanoPeriGPT training script — adapted from nanoGPT for MPS/CPU and peridynamic configs.

Usage:
    python train.py config/baseline.py
    python train.py config/baseline.py --attention_type=peridynamic --horizon=32
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = 'out'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
# wandb logging
wandb_log = False
wandb_project = 'nanoPeriGPT'
wandb_run_name = 'run'
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
# model (vanilla nanoGPT defaults)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False
# peridynamic extensions (declared here so configurator can override them)
attention_type = 'standard'
horizon = 32
n_global_anchors = 8
bond_dim_ratio = 4
residual_type = 'standard'
depth_block_size = 4
block_damage = False
deer_enabled = False
deer_method = 'quasi-deer'
deer_max_iter = 10
deer_tol = 1e-4
deer_damping = 0.3
deer_damage_scale = 1.0
deer_warmstart = 'picard'
energy_lambda = 0.0
# optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
# lr decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
# system — auto-detect best device
if torch.cuda.is_available():
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    compile = True
elif torch.backends.mps.is_available():
    device = 'mps'
    dtype = 'float32'
    compile = False
else:
    device = 'cpu'
    dtype = 'float32'
    compile = False
# experiment tracking
experiment_name = ''  # set via config or auto-generated
# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items()
    if not k.startswith('_') and isinstance(v, (int, float, bool, str))
]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# auto-generate experiment name from config if not set
if not experiment_name:
    experiment_name = wandb_run_name if wandb_run_name != 'run' else out_dir.replace('out-', '')

master_process = True  # no DDP — single device
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

device_type = 'cpu'
if 'cuda' in device:
    device_type = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif 'mps' in device:
    device_type = 'mps'

ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}[dtype]

if device_type == 'cpu' or device_type == 'mps':
    ctx = nullcontext()
else:
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# data loader
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)


def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# model init
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# build model_args — include all GPTConfig fields
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    # peridynamic extensions
    attention_type=attention_type,
    horizon=horizon,
    n_global_anchors=n_global_anchors,
    bond_dim_ratio=bond_dim_ratio,
    residual_type=residual_type,
    depth_block_size=depth_block_size,
    block_damage=block_damage,
    deer_enabled=deer_enabled,
    deer_method=deer_method,
    deer_max_iter=deer_max_iter,
    deer_tol=deer_tol,
    deer_damping=deer_damping,
    deer_damage_scale=deer_damage_scale,
    deer_warmstart=deer_warmstart,
    energy_lambda=energy_lambda,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# GradScaler only for CUDA float16
scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda' and dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

print(f"device: {device}, dtype: {dtype}, compile: {compile}")
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")
print(f"config: attention_type={attention_type}, horizon={horizon}, "
      f"residual_type={residual_type}, deer_enabled={deer_enabled}")


# -----------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------------------------------------------------------
# lr schedule
# -----------------------------------------------------------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# wandb
# -----------------------------------------------------------------------------
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
final_train_loss = None
final_val_loss = None

while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        final_train_loss = losses['train'].item()
        final_val_loss = losses['val'].item()
        print(f"step {iter_num}: train loss {final_train_loss:.4f}, val loss {final_val_loss:.4f}")
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            }
            wandb.log(log_dict)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

# -----------------------------------------------------------------------------
# final evaluation and results logging
# -----------------------------------------------------------------------------
losses = estimate_loss()
final_train_loss = losses['train'].item()
final_val_loss = losses['val'].item()
print(f"\n=== FINAL: train loss {final_train_loss:.4f}, val loss {final_val_loss:.4f} ===")

# append to results.tsv
results_file = 'results.tsv'
header_needed = not os.path.exists(results_file)
with open(results_file, 'a') as f:
    if header_needed:
        f.write("experiment\tval_loss\ttrain_loss\tstatus\tdescription\n")
    desc = (f"attn={attention_type} horizon={horizon} res={residual_type} "
            f"deer={deer_enabled} energy_lambda={energy_lambda} "
            f"n_layer={n_layer} n_head={n_head} n_embd={n_embd}")
    f.write(f"{experiment_name}\t{final_val_loss:.4f}\t{final_train_loss:.4f}\tKEEP\t{desc}\n")

print(f"Results appended to {results_file}")
