# Config for the FULL unified nanoPeriGPT:
#   - Peridynamic attention (bond-based, sequence dimension)
#   - Block Attention Residual (state-based, depth dimension)
#   - DEER layer-parallel inference (multigrid via block structure)
#
# This is the "everything connected" configuration.
#
# Usage:
#   python train.py config/train_shakespeare_unified.py

# --- attention type (sequence dimension: bond-based peridynamics) ---
attention_type = 'peridynamic'
horizon = 32

# --- residual type (depth dimension: state-based peridynamics) ---
residual_type = 'block_attn'
depth_block_size = 3   # 6 layers / 3 = 2 blocks

# --- DEER (enabled at inference) ---
deer_enabled = True
deer_method = 'damage-elk'
deer_max_iter = 10
deer_tol = 1e-4
deer_damping = 0.3
deer_damage_scale = 2.0
deer_warmstart = 'picard'

# --- model ---
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.2
bias = False

# --- training ---
out_dir = 'out-shakespeare-unified'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanoPeriGPT'
wandb_run_name = 'unified-peri-blockres-deer'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
max_iters = 5000
lr_decay_iters = 5000

learning_rate = 1e-3
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
