#!/usr/bin/env bash
# Warm-start finetune of sonic G1 on the 1000-clip set (Intel XPU), WITH the official
# critic warm-loaded from last.pt (value_state_dict) -> NO cold critic.
#
# vs run_warm_1k_mixed.sh: that script predates critic warm-load, so it had to start a
# COLD critic and lean on cold-critic mitigations (CRITIC_WARMUP=30 to burn the fresh
# critic in first, SAMPLING_MODE=mixed to avoid the adaptive "hardest-frame" trap while
# the critic was unusable, TARGET_KL_STOP=0.025 to cap OOD drift, and a tiny actor
# LR=5e-6 with a fast CRITIC_LR=2e-4 so the fresh critic could sprint). Now that the critic
# loads the official weights + RunningMeanStd from last.pt (env.critic_privileged_mf_hist
# =true -> 1645-d value obs; see sonic.yaml), none of those crutches are needed.
#
# LR REGIME = native sonic finetune (gear_sonic config/algo/ppo_im_phc.yaml), verified in
# source: the trainer's LR is wired ONLY to actor_learning_rate=2e-5
# (trl/ppo.yaml: learning_rate=${algo.config.actor_learning_rate}); critic_learning_rate=1e-3
# is DEFINED BUT NEVER REFERENCED (dead param) — official uses ONE SHARED LR across actor+
# critic, adjusted by adaptive-KL (schedule=adaptive, /1.5 or *1.5, clamp [1e-5, 2e-4],
# desired_kl=0.01). So we DO NOT set CRITIC_LR here: a split LR would pin schedule="fixed"
# and kill the adaptive-KL behavior official relies on. Instead: shared LR=2e-5 (initial) +
# ADAPTIVE_LR_MAX=2e-4 (match the official ceiling; UniLab's default null would let the
# adaptive schedule climb to rsl_rl's 1e-2). desired_kl=0.01 is already the config default.
# Dropped vs mixed: CRITIC_WARMUP (critic not fresh), TARGET_KL_STOP (adaptive schedule
# handles it), SAMPLING_MODE (reverts to native sonic "adaptive" sampling).
#
# Launches detached via setsid+nohup so it survives the terminal closing.
set -euo pipefail

UNILAB=/home/maozhong/work/my_sonic/UniLab
cd "$UNILAB"

mkdir -p /tmp/sonic_train
LOG=/tmp/sonic_train/warm_1k_sonic_critic_$(date +%Y%m%d_%H%M%S).log

# CKPT = actor + critic warm-start source (CRITIC_CKPT defaults to CKPT in train_warm.sh).
# FREEZE_ENCODER=true keeps the FSQ token space fixed (VLA-compat); set false to finetune
# the encoder too. To A/B against a fresh critic: append CRITIC_CKPT=null (then optionally
# re-add CRITIC_WARMUP / SAMPLING_MODE=mixed as in run_warm_1k_mixed.sh).
NPZ_DIR=/home/maozhong/work/sonic_vla_infer/bones_seed_1k/npz \
CKPT=/home/maozhong/work/my_sonic/UniLab/last.pt \
CRITIC_CKPT=/home/maozhong/work/my_sonic/UniLab/last.pt \
NUM_ENVS=4096 ITERS=5000 SAVE=200 SAMPLING_MODE=mixed \
FREEZE_ENCODER=true LR=0.00002 ADAPTIVE_LR_MAX=2e-4 \
setsid nohup ./scripts/sonic/train_warm.sh run > "$LOG" 2>&1 < /dev/null &

echo "launched (official critic warm-loaded from last.pt). log: $LOG"
echo "monitor:  tail -f $LOG"
echo "stop:     pkill -f 'train_rsl_rl.py.*pretrained_ckpt'"
