# GPT-2 Small (125M) on SlimPajama GitHub-only
# Code is naturally heterogeneous — no synthetic mixing needed.

out_dir = 'out-gh-baseline'
experiment_name = 'gh_baseline'

dataset = 'slimpajama_github'
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 4  # 32*1024*4 = 131K tokens/step (smaller dataset)

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

attention_type = 'standard'
residual_type = 'standard'
horizon = 128
bond_dim_ratio = 2
depth_block_size = 4
block_damage = False
deer_enabled = False
energy_lambda = 0.0

learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

max_iters = 3000      # 131K * 3K = ~393M tokens
lr_decay_iters = 3000
min_lr = 6e-5
warmup_iters = 300
decay_lr = True

eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
wandb_log = False
wandb_project = 'perigpt-github'
wandb_run_name = 'gh_baseline'
