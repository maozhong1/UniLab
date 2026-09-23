#!/usr/bin/env bash
set -euo pipefail
# H2 SONIC full-train — CURRICULUM stage on the CALM motion subset.
# Rationale: motion_body_ang_vel was pinned at its "robot barely rotates" floor (~0.16)
# because the full clip set has reference body ang-vel up to 8-19 rad/s that a torque-
# limited humanoid cannot reproduce upright. Removing the unfollowable spins lets the
# policy actually track rotation and stabilizes ep_len. Once this subset trains well,
# widen the threshold (rebuild with a larger --thresh) as a curriculum.
#   * keep condim=6 foot contact (h2.xml) + feet_slide/feet_yaw penalties (stability).

UNILAB=/home/maozhong/work/my_sonic/UniLab
cd "$UNILAB"

mkdir -p /tmp/sonic_train
LOG=/tmp/sonic_train/full_train_h2_calm_$(date +%Y%m%d_%H%M%S).log

# Resume the newest checkpoint of the run we are continuing.
LOAD_RUN=2026-09-20_14-58-44_mujoco
RUN_DIR="$UNILAB/logs/rsl_rl_ppo/H2SonicMotionTracking/$LOAD_RUN"
CKPT=$(ls "$RUN_DIR"/model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
if [[ -z "${CKPT:-}" ]]; then echo "ERROR: no model_*.pt in $RUN_DIR" >&2; exit 1; fi
echo "resuming $LOAD_RUN @ checkpoint $CKPT (calm subset)"

# Calm subset produced by:
#   uv run --no-sync python scripts/motion/filter_calm_h2.py \
#     --src .../npz_h2_new --dst .../npz_h2_calm --metric p99 --thresh 8.0
NPZ_DIR=/home/maozhong/work/sonic_vla_infer/bones_h2_calm_3k
mapfile -t NPZ_FILES < <(ls "$NPZ_DIR"/*.npz 2>/dev/null)
if [[ ${#NPZ_FILES[@]} -eq 0 ]]; then echo "ERROR: no npz in $NPZ_DIR (run filter_calm_h2.py first)" >&2; exit 1; fi

# Hydra's CLI parser rejects any single argument over ~128KB (kernel MAX_ARG_STRLEN);
# with 3000+ absolute npz paths, an inline "+env.motion_file=[...]" override blows past
# that ("Argument list too long"). Write the list to a config file instead and merge it
# via hydra.searchpath so the giant list never touches argv.
MOTION_OVERRIDE_ROOT=/tmp/sonic_train
MOTION_OVERRIDE_NAME=h2_3k
mkdir -p "$MOTION_OVERRIDE_ROOT/motion_override"
MOTION_OVERRIDE_FILE="$MOTION_OVERRIDE_ROOT/motion_override/$MOTION_OVERRIDE_NAME.yaml"
{
  echo "# @package _global_"
  echo "env:"
  echo "  motion_file:"
  for f in "${NPZ_FILES[@]}"; do echo "    - $f"; done
} > "$MOTION_OVERRIDE_FILE"
echo "wrote ${#NPZ_FILES[@]} motion clips -> $MOTION_OVERRIDE_FILE"

NUM_ENVS=4096 ; NUM_STEPS=24

export HF_ENDPOINT=https://hf-mirror.com

setsid nohup uv run --no-sync python scripts/train_rsl_rl.py \
  task=h2_motion_tracking/sonic_full_train training.device=xpu training.no_play=true \
  algo.actor.distribution_cfg.init_std=0.48 \
  algo.resume=true \
  algo.load_run="$LOAD_RUN" \
  algo.checkpoint="$CKPT" \
  algo.algorithm.learning_rate=2e-5 \
  algo.algorithm.entropy_coef=0.004 \
  algo.algorithm.desired_kl=0.01 \
  algo.algorithm.adaptive_lr_max=2e-4 \
  algo.num_steps_per_env="$NUM_STEPS" \
  reward.scales.motion_body_ang_vel=1.0 \
  reward.scales.motion_body_ori=1.5 \
  reward.scales.feet_slide=-0.8 \
  reward.scales.feet_yaw=-0.1 \
  reward.scales.action_rate_l2=-0.08 \
  "+env.critic_privileged_mf_hist=true" \
  "hydra.searchpath=[file://$MOTION_OVERRIDE_ROOT]" \
  "+motion_override=$MOTION_OVERRIDE_NAME" \
  "+env.sampling_mode=mixed" \
  algo.num_envs="$NUM_ENVS" algo.max_iterations=30000  algo.save_interval=100 > "$LOG" 2>&1 < /dev/null &

echo "launched H2 CALM-curriculum RESUME @${CKPT} (${#NPZ_FILES[@]} clips, ang_vel=1.0), log=$LOG"
