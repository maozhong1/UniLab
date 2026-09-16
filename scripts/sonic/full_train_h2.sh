#!/usr/bin/env bash
set -euo pipefail
# H2 (31-DOF) from-scratch SONIC full-train. Forked from full_train.sh (G1).
#   task:              h2_motion_tracking/sonic_full_train (H2SonicMotionTracking env)
#   actor:             SonicH2ActorModel, obs 1670 = enc(680)+proprio(990), action 31
#   critic:            1745 privileged_mf_hist (+env.critic_privileged_mf_hist=true), fresh
#   data:              H2 npz converted from G1 CSV via scripts/motion/g1_csv_to_h2_npz.py
#   num_envs x steps:  4096 x 24 (official batch); LR 2e-5..2e-4 kl0.01, init_std 0.5

UNILAB=/home/maozhong/work/my_sonic/UniLab
cd "$UNILAB"

mkdir -p /tmp/sonic_train
LOG=/tmp/sonic_train/full_train_h2_$(date +%Y%m%d_%H%M%S).log
# H2 npz dir (produced by g1_csv_to_h2_npz.py). Change to your prepared set.
NPZ_DIR=/home/maozhong/work/sonic_vla_infer/bones_seed_1k/npz_h2
LIST=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | paste -sd, -)

NUM_ENVS=4096 ; NUM_STEPS=24

export HF_ENDPOINT=https://hf-mirror.com

setsid nohup uv run --no-sync python scripts/train_rsl_rl.py \
  task=h2_motion_tracking/sonic_full_train training.device=xpu training.no_play=true \
  algo.actor.distribution_cfg.init_std=0.5 \
  algo.algorithm.learning_rate=2e-5 \
  algo.algorithm.entropy_coef=0.004 \
  algo.algorithm.desired_kl=0.01 \
  algo.algorithm.adaptive_lr_max=2e-4 \
  algo.num_steps_per_env="$NUM_STEPS" \
  "+env.critic_privileged_mf_hist=true" \
  "+env.motion_file=[$LIST]" \
  "+env.sampling_mode=mixed" \
  algo.num_envs="$NUM_ENVS" algo.max_iterations=10000  algo.save_interval=100 > "$LOG" 2>&1 < /dev/null &

echo "launched H2 full-train (actor 1670->31, critic 1745 privileged, ${NUM_ENVS}x${NUM_STEPS}), log=$LOG"
