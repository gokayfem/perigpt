# GPT-2 Small (125M) on SlimPajama — standard attention baseline
# Standard recipe from nanoGPT for OpenWebText, adapted for SlimPajama.

out_dir = 'out-sp-baseline'
experiment_name = 'sp_baseline'

# data
dataset = 'slimpajama'
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 8  # 32 * 1024 * 8 = 262K tokens/step

# model — GPT-2 Small
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# standard attention
attention_type = 'standard'
residual_type = 'standard'
horizon = 128
bond_dim_ratio = 2
depth_block_size = 4
block_damage = False
deer_enabled = False
energy_lambda = 0.0

# optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# schedule
max_iters = 10000     # ~2.6B tokens
lr_decay_iters = 10000
min_lr = 6e-5
warmup_iters = 500
decay_lr = True

# eval
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

# logging
wandb_log = False
wandb_project = 'perigpt-slimpajama'
wandb_run_name = 'sp_baseline'
