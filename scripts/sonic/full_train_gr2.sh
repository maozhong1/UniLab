#!/usr/bin/env bash
set -euo pipefail

# GR2 SONIC curriculum training for the fixed-head 27-DoF profile.
# Stage 1 starts from scratch. Stages 2, 3, and full require LOAD_RUN and
# resume its newest checkpoint. Set NPZ_DIR to bypass automatic filtering.

UNILAB="${UNILAB:-/home/maozhong/work/my_sonic/UniLab}"
SOURCE_NPZ_DIR="${SOURCE_NPZ_DIR:-$HOME/work/sonic_vla_infer/bones_seed_3k_new/npz_gr2_27dof}"
CURRICULUM_STAGE="${CURRICULUM_STAGE:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_STEPS="${NUM_STEPS:-24}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
WORK_DIR="${WORK_DIR:-/tmp/sonic_train}"

cd "$UNILAB"
mkdir -p "$WORK_DIR/motion_override"

if [[ -z "${NPZ_DIR:-}" ]]; then
  case "$CURRICULUM_STAGE" in
    1)
      FILTER_ARGS=(
        --body-ang-p99 6.5 --joint-vel-p99 5.5
        --base-lin-p99 1.35 --base-ang-p99 2.0
        --base-z-range 0.15 --foot-height-max 0.35
      )
      ;;
    2)
      FILTER_ARGS=(
        --body-ang-p99 8.0 --joint-vel-p99 7.0
        --base-lin-p99 1.8 --base-ang-p99 2.5
        --base-z-range 0.25 --foot-height-max 0.45
      )
      ;;
    3)
      FILTER_ARGS=(
        --body-ang-p99 12.0 --joint-vel-p99 9.5
        --base-lin-p99 2.2 --base-ang-p99 4.0
        --base-z-range 0.35 --foot-height-max 0.65
      )
      ;;
    full)
      FILTER_ARGS=()
      ;;
    *)
      echo "ERROR: CURRICULUM_STAGE must be 1, 2, 3, or full" >&2
      exit 2
      ;;
  esac

  if [[ "$CURRICULUM_STAGE" == "full" ]]; then
    NPZ_DIR="$SOURCE_NPZ_DIR"
  else
    NPZ_DIR="$WORK_DIR/gr2_stage${CURRICULUM_STAGE}"
    uv run --no-sync python scripts/motion/filter_calm_gr2.py \
      --src "$SOURCE_NPZ_DIR" --dst "$NPZ_DIR" --show-dropped 0 \
      "${FILTER_ARGS[@]}"
  fi
fi

shopt -s nullglob
NPZ_FILES=("$NPZ_DIR"/*.npz)
shopt -u nullglob
if [[ ${#NPZ_FILES[@]} -eq 0 ]]; then
  echo "ERROR: no NPZ clips in $NPZ_DIR" >&2
  exit 1
fi

# Keep the 3000+ paths out of argv (Linux MAX_ARG_STRLEN) by composing a
# temporary Hydra config through hydra.searchpath.
MOTION_OVERRIDE_NAME="gr2_stage_${CURRICULUM_STAGE}_$$"
MOTION_OVERRIDE_FILE="$WORK_DIR/motion_override/$MOTION_OVERRIDE_NAME.yaml"
{
  echo "# @package _global_"
  echo "env:"
  echo "  motion_file:"
  for path in "${NPZ_FILES[@]}"; do
    printf '    - %s\n' "$path"
  done
} > "$MOTION_OVERRIDE_FILE"
echo "wrote ${#NPZ_FILES[@]} motion clips -> $MOTION_OVERRIDE_FILE"

RESUME_ARGS=(algo.resume=false)
if [[ "$CURRICULUM_STAGE" != "1" ]]; then
  if [[ -z "${LOAD_RUN:-}" ]]; then
    echo "ERROR: LOAD_RUN is required for curriculum stage $CURRICULUM_STAGE" >&2
    exit 2
  fi
  RUN_DIR="$UNILAB/logs/rsl_rl_ppo/GR2SonicMotionTracking/$LOAD_RUN"
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: GR2 run directory does not exist: $RUN_DIR" >&2
    exit 1
  fi
  CKPT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
    | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
  if [[ -z "${CKPT:-}" ]]; then
    echo "ERROR: no model_*.pt in $RUN_DIR" >&2
    exit 1
  fi
  RESUME_ARGS=(algo.resume=true "algo.load_run=$LOAD_RUN" "algo.checkpoint=$CKPT")
  echo "resuming $LOAD_RUN @ checkpoint $CKPT"
elif [[ -n "${LOAD_RUN:-}" ]]; then
  echo "ERROR: Stage 1 is from scratch; unset LOAD_RUN" >&2
  exit 2
fi

TRAIN_ARGS=(
  task=gr2_motion_tracking/sonic_full_train
  training.device=xpu
  training.no_play=true
  algo.actor.distribution_cfg.init_std=0.50
  "${RESUME_ARGS[@]}"
  algo.algorithm.learning_rate=2e-5
  algo.algorithm.entropy_coef=0.004
  algo.algorithm.desired_kl=0.01
  algo.algorithm.adaptive_lr_max=2e-4
  "algo.num_steps_per_env=$NUM_STEPS"
  reward.scales.joint_acc_l2=-2.5e-7
  reward.scales.joint_torque_l2=-1.0e-6
  reward.scales.anti_shake_ang_vel=-2.5e-3
  env.critic_privileged_mf_hist=true
  "hydra.searchpath=[file://$WORK_DIR]"
  "+motion_override=$MOTION_OVERRIDE_NAME"
  "++env.sampling_mode=mixed"
  "algo.num_envs=$NUM_ENVS"
  "algo.max_iterations=$MAX_ITERATIONS"
  "algo.save_interval=$SAVE_INTERVAL"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  uv run --no-sync python scripts/train_rsl_rl.py "${TRAIN_ARGS[@]}" --cfg job >/dev/null
  echo "Hydra compose passed (stage=$CURRICULUM_STAGE, clips=${#NPZ_FILES[@]})"
  exit 0
fi

LOG="$WORK_DIR/full_train_gr2_stage${CURRICULUM_STAGE}_$(date +%Y%m%d_%H%M%S).log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
setsid nohup uv run --no-sync python scripts/train_rsl_rl.py "${TRAIN_ARGS[@]}" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched GR2 stage $CURRICULUM_STAGE (${#NPZ_FILES[@]} clips), log=$LOG"