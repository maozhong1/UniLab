#!/usr/bin/env bash
set -euo pipefail
# EXP1+2 (align critic + LR to official SONIC from-scratch):
#   critic:            926 (286 privileged + 640 mf ref) -> [2048,2048,1024,1024,512,512]
#                      SiLU (was [1024,1024,512,512]); set in sonic_full_train.yaml
#   LR regime:         shared adaptive, init 2e-5, floor 1e-5, ceiling 2e-4, desired_kl 0.01
#                      (official ppo_im_phc.yaml; was 1e-3 init / 1e-2 ceiling / kl 0.005)
#   num_envs x steps:  4096 x 24 = 98304 (official batch; XPU-probed, ~17s/iter)
#   init_std 0.5, entropy_coef 0.005, action_rate_l2 -0.01 (loose bootstrap, anneal later)

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
  algo.algorithm.learning_rate=2e-5 \
  algo.algorithm.entropy_coef=0.004 \
  algo.algorithm.desired_kl=0.01 \
  algo.algorithm.adaptive_lr_max=2e-4 \
  algo.num_steps_per_env="$NUM_STEPS" \
  "+env.critic_privileged_mf_hist=true" \
  "+env.motion_file=[$LIST]" \
  "+env.sampling_mode=mixed" \
  algo.num_envs="$NUM_ENVS" algo.max_iterations=10000  algo.save_interval=100 > "$LOG" 2>&1 < /dev/null &

echo "launched EXP1+2 run (critic 926->[2048,2048,1024,1024,512,512] SiLU, LR 2e-5..2e-4 kl0.01, ${NUM_ENVS}x${NUM_STEPS}), log=$LOG"
