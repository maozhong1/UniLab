#!/usr/bin/env bash
set -euo pipefail
# Config = the proven healthy fresh recipe + the validated 4096x24 official batch:
#   critic:            926 (286 privileged + 640 future ref) -> [1024,1024,512,512] SiLU
#                      (set in sonic.yaml; env.critic_include_future=true by default)
#   num_envs x steps:  4096 x 24 = 98304 (official batch; XPU-probed, ~17s/iter)
#   init_std 0.5, entropy_coef 0.005, action_rate_l2 -0.01 (from sonic.yaml; loose for
#                      bootstrap, anneal to -0.05 later), desired_kl 0.005

UNILAB=/home/maozhong/work/my_sonic/UniLab
cd "$UNILAB"

mkdir -p /tmp/sonic_train
LOG=/tmp/sonic_train/full_train_$(date +%Y%m%d_%H%M%S).log
#change to your prepared npz file
NPZ_DIR=/home/maozhong/work/sonic_vla_infer/bones_seed_1k/npz
LIST=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | paste -sd, -)

NUM_ENVS=4096 ; NUM_STEPS=24

export HF_ENDPOINT=https://hf-mirror.com

setsid nohup uv run --no-sync python scripts/train_rsl_rl.py \
  task=g1_motion_tracking/sonic_full_train training.device=xpu training.no_play=true \
  algo.actor.distribution_cfg.init_std=0.5 \
  algo.algorithm.learning_rate=0.001 \
  algo.algorithm.entropy_coef=0.005 \
  algo.algorithm.desired_kl=0.005 \
  algo.num_steps_per_env="$NUM_STEPS" \
  "+env.motion_file=[$LIST]" \
  algo.num_envs="$NUM_ENVS" algo.max_iterations=10000 algo.save_interval=500 > "$LOG" 2>&1 < /dev/null &

echo "launched full fresh run (critic 926 SiLU, ${NUM_ENVS}x${NUM_STEPS}), log=$LOG"
