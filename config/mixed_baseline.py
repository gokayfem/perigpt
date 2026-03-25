# Mixed-domain baseline: standard attention on interleaved multi-domain data
out_dir = 'out-mixed-baseline'
wandb_run_name = 'mixed_baseline'
experiment_name = 'mixed_baseline'

dataset = 'mixed_domain'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

attention_type = 'standard'
horizon = 32
residual_type = 'standard'
depth_block_size = 3
deer_enabled = False
energy_lambda = 0.0

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
wandb_log = False
