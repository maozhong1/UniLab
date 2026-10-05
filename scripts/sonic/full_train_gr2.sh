#!/usr/bin/env bash
set -euo pipefail

# GR2 SONIC curriculum training for the fixed-head 27-DoF profile.
# Stage 1 starts from scratch. Later stages require LOAD_RUN and resume its newest
# checkpoint unless CHECKPOINT is set. Stages 2.5 and 2.8 merge manipulation clips
# into the base dataset before filtering. Set NPZ_DIR to bypass filtering.

UNILAB="${UNILAB:-/home/maozhong/work/my_sonic/UniLab}"
SOURCE_NPZ_DIR="${SOURCE_NPZ_DIR:-$HOME/work/sonic_vla_infer/bones_seed_3k_new/npz_gr2_27dof}"
MANIPULATION_NPZ_DIR="${MANIPULATION_NPZ_DIR:-$HOME/work/sonic_vla_infer/bones-seed/selections/gr2_upright_operations_npz}"
CURRICULUM_STAGE="${CURRICULUM_STAGE:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_STEPS="${NUM_STEPS:-24}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
WORK_DIR="${WORK_DIR:-/tmp/sonic_train}"

if [[ "$CURRICULUM_STAGE" == "2.8" ]]; then
  DEFAULT_LEARNING_RATE=7.5e-6
  DEFAULT_ADAPTIVE_LR_MAX=1e-5
  DEFAULT_ENTROPY_COEF=0.001
  DEFAULT_ROOT_POS_SCALE=0.75
  DEFAULT_ACTION_RATE_L2_SCALE=0.0
  DEFAULT_TARGET_RATE_L2_SCALE=-0.02
  DEFAULT_UPPER_BODY_JOINT_POS_SCALE=1.0
  DEFAULT_ANTI_SHAKE_ANG_VEL_SCALE=0.0
  DEFAULT_FOCUS_PATTERN='*macarena*.npz'
  DEFAULT_FOCUS_REPEAT=5
elif [[ "$CURRICULUM_STAGE" == "2.5" ]]; then
  DEFAULT_LEARNING_RATE=1e-5
  DEFAULT_ADAPTIVE_LR_MAX=2e-4
  DEFAULT_ENTROPY_COEF=0.003
  DEFAULT_ROOT_POS_SCALE=0.75
  DEFAULT_ACTION_RATE_L2_SCALE=0.0
  DEFAULT_TARGET_RATE_L2_SCALE=-0.05
  DEFAULT_UPPER_BODY_JOINT_POS_SCALE=0.5
  DEFAULT_ANTI_SHAKE_ANG_VEL_SCALE=-2.5e-3
  DEFAULT_FOCUS_PATTERN=''
  DEFAULT_FOCUS_REPEAT=1
else
  DEFAULT_LEARNING_RATE=2e-5
  DEFAULT_ADAPTIVE_LR_MAX=2e-4
  DEFAULT_ENTROPY_COEF=0.004
  DEFAULT_ROOT_POS_SCALE=0.5
  DEFAULT_ACTION_RATE_L2_SCALE=-0.01
  DEFAULT_TARGET_RATE_L2_SCALE=0.0
  DEFAULT_UPPER_BODY_JOINT_POS_SCALE=0.0
  DEFAULT_ANTI_SHAKE_ANG_VEL_SCALE=-2.5e-3
  DEFAULT_FOCUS_PATTERN=''
  DEFAULT_FOCUS_REPEAT=1
fi
LEARNING_RATE="${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
ENTROPY_COEF="${ENTROPY_COEF:-$DEFAULT_ENTROPY_COEF}"
ADAPTIVE_LR_MAX="${ADAPTIVE_LR_MAX:-$DEFAULT_ADAPTIVE_LR_MAX}"
ROOT_POS_SCALE="${ROOT_POS_SCALE:-$DEFAULT_ROOT_POS_SCALE}"
ACTION_RATE_L2_SCALE="${ACTION_RATE_L2_SCALE:-$DEFAULT_ACTION_RATE_L2_SCALE}"
TARGET_RATE_L2_SCALE="${TARGET_RATE_L2_SCALE:-$DEFAULT_TARGET_RATE_L2_SCALE}"
UPPER_BODY_JOINT_POS_SCALE="${UPPER_BODY_JOINT_POS_SCALE:-$DEFAULT_UPPER_BODY_JOINT_POS_SCALE}"
ANTI_SHAKE_ANG_VEL_SCALE="${ANTI_SHAKE_ANG_VEL_SCALE:-$DEFAULT_ANTI_SHAKE_ANG_VEL_SCALE}"
FOCUS_PATTERN="${FOCUS_PATTERN:-$DEFAULT_FOCUS_PATTERN}"
FOCUS_REPEAT="${FOCUS_REPEAT:-$DEFAULT_FOCUS_REPEAT}"
INIT_STD="${INIT_STD:-0.50}"

if [[ "$CURRICULUM_STAGE" == "3" ]]; then
  DEFAULT_STRICT_FOOT_POS_THRESHOLD=0.45
  DEFAULT_STRICT_ANCHOR_ORI_ERROR_SQ=0.70
  DEFAULT_STRICT_HEIGHT_THRESHOLD=0.40
else
  DEFAULT_STRICT_FOOT_POS_THRESHOLD=0.50
  DEFAULT_STRICT_ANCHOR_ORI_ERROR_SQ=0.80
  DEFAULT_STRICT_HEIGHT_THRESHOLD=0.45
fi
STRICT_FOOT_POS_THRESHOLD="${STRICT_FOOT_POS_THRESHOLD:-$DEFAULT_STRICT_FOOT_POS_THRESHOLD}"
STRICT_ANCHOR_ORI_ERROR_SQ="${STRICT_ANCHOR_ORI_ERROR_SQ:-$DEFAULT_STRICT_ANCHOR_ORI_ERROR_SQ}"
STRICT_HEIGHT_THRESHOLD="${STRICT_HEIGHT_THRESHOLD:-$DEFAULT_STRICT_HEIGHT_THRESHOLD}"

if [[ ! "$FOCUS_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: FOCUS_REPEAT must be a positive integer" >&2
  exit 2
fi

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
    2|2.5)
      FILTER_ARGS=(
        --body-ang-p99 8.0 --joint-vel-p99 7.0
        --base-lin-p99 1.8 --base-ang-p99 2.5
        --base-z-range 0.25 --foot-height-max 0.45
      )
      ;;
    2.8)
      FILTER_ARGS=(
        --body-ang-p99 10.0 --joint-vel-p99 8.5
        --base-lin-p99 2.0 --base-ang-p99 3.5
        --base-z-range 0.30 --foot-height-max 0.55
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
      echo "ERROR: CURRICULUM_STAGE must be 1, 2, 2.5, 2.8, 3, or full" >&2
      exit 2
      ;;
  esac

  if [[ "$CURRICULUM_STAGE" == "full" ]]; then
    NPZ_DIR="$SOURCE_NPZ_DIR"
  else
    FILTER_SOURCE_DIR="$SOURCE_NPZ_DIR"
    STAGE_DIR_SUFFIX="${CURRICULUM_STAGE//./_}"
    if [[ "$CURRICULUM_STAGE" == "2.5" || "$CURRICULUM_STAGE" == "2.8" ]]; then
      FILTER_SOURCE_DIR="$WORK_DIR/gr2_stage${STAGE_DIR_SUFFIX}_source"
      mkdir -p "$FILTER_SOURCE_DIR"
      for existing in "$FILTER_SOURCE_DIR"/*.npz; do
        [[ -e "$existing" || -L "$existing" ]] || continue
        if [[ ! -L "$existing" ]]; then
          echo "ERROR: refusing to replace real file in $FILTER_SOURCE_DIR: $existing" >&2
          exit 1
        fi
        rm "$existing"
      done
      MERGED_CLIPS=0
      for source_dir in "$SOURCE_NPZ_DIR" "$MANIPULATION_NPZ_DIR"; do
        if [[ ! -d "$source_dir" ]]; then
          echo "ERROR: motion source directory does not exist: $source_dir" >&2
          exit 1
        fi
        SOURCE_CLIPS=("$source_dir"/*.npz)
        if [[ ! -e "${SOURCE_CLIPS[0]}" ]]; then
          echo "ERROR: no NPZ clips in motion source: $source_dir" >&2
          exit 1
        fi
        for source in "${SOURCE_CLIPS[@]}"; do
          target="$FILTER_SOURCE_DIR/${source##*/}"
          if [[ -e "$target" || -L "$target" ]]; then
            echo "ERROR: duplicate NPZ filename while merging Stage $CURRICULUM_STAGE: ${source##*/}" >&2
            exit 1
          fi
          ln -s "$source" "$target"
          ((++MERGED_CLIPS))
        done
      done
      echo "merged $MERGED_CLIPS motion clips -> $FILTER_SOURCE_DIR"
    fi
    NPZ_DIR="$WORK_DIR/gr2_stage${STAGE_DIR_SUFFIX}"
    uv run --no-sync python scripts/motion/filter_calm_gr2.py \
      --src "$FILTER_SOURCE_DIR" --dst "$NPZ_DIR" --show-dropped 0 \
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

if [[ -n "$FOCUS_PATTERN" && "$FOCUS_REPEAT" -gt 1 ]]; then
  FOCUS_FILES=()
  for path in "${NPZ_FILES[@]}"; do
    if [[ "${path##*/}" == $FOCUS_PATTERN ]]; then
      FOCUS_FILES+=("$path")
    fi
  done
  if [[ ${#FOCUS_FILES[@]} -eq 0 ]]; then
    echo "ERROR: FOCUS_PATTERN matched no NPZ clips: $FOCUS_PATTERN" >&2
    exit 1
  fi
  UNIQUE_FRAME_COUNT=$(uv run --no-sync python - "${NPZ_FILES[@]}" <<'PY'
import sys

import numpy as np

print(sum(np.load(path, mmap_mode="r")["joint_pos"].shape[0] for path in sys.argv[1:]))
PY
  )
  FOCUS_FRAME_COUNT=$(uv run --no-sync python - "${FOCUS_FILES[@]}" <<'PY'
import sys

import numpy as np

print(sum(np.load(path, mmap_mode="r")["joint_pos"].shape[0] for path in sys.argv[1:]))
PY
  )
  for ((repeat = 1; repeat < FOCUS_REPEAT; repeat++)); do
    NPZ_FILES+=("${FOCUS_FILES[@]}")
  done
  EFFECTIVE_FOCUS_PERCENT=$(uv run --no-sync python - <<PY
focus = $FOCUS_FRAME_COUNT * $FOCUS_REPEAT
total = $UNIQUE_FRAME_COUNT + $FOCUS_FRAME_COUNT * ($FOCUS_REPEAT - 1)
print(f"{100.0 * focus / total:.2f}")
PY
  )
  echo "oversampled ${#FOCUS_FILES[@]} focus clips ${FOCUS_REPEAT}x ($FOCUS_PATTERN), effective frame share ${EFFECTIVE_FOCUS_PERCENT}%"
fi

# Keep the 3000+ paths out of argv (Linux MAX_ARG_STRLEN) by composing a
# temporary Hydra config through hydra.searchpath.
MOTION_OVERRIDE_NAME="gr2_stage_${CURRICULUM_STAGE//./_}_$$"
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
  if [[ -n "${CHECKPOINT:-}" ]]; then
    CKPT="${CHECKPOINT#model_}"
    CKPT="${CKPT%.pt}"
    if [[ ! "$CKPT" =~ ^[0-9]+$ || ! -f "$RUN_DIR/model_$CKPT.pt" ]]; then
      echo "ERROR: checkpoint does not exist: $RUN_DIR/model_$CKPT.pt" >&2
      exit 1
    fi
  else
    CKPT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
      | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
  fi
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
  "algo.actor.distribution_cfg.init_std=$INIT_STD"
  "${RESUME_ARGS[@]}"
  "algo.algorithm.learning_rate=$LEARNING_RATE"
  "algo.algorithm.entropy_coef=$ENTROPY_COEF"
  algo.algorithm.desired_kl=0.01
  "algo.algorithm.adaptive_lr_max=$ADAPTIVE_LR_MAX"
  "algo.num_steps_per_env=$NUM_STEPS"
  "reward.scales.motion_global_root_pos=$ROOT_POS_SCALE"
  "reward.scales.action_rate_l2=$ACTION_RATE_L2_SCALE"
  "reward.scales.target_rate_l2=$TARGET_RATE_L2_SCALE"
  "reward.scales.upper_body_joint_pos=$UPPER_BODY_JOINT_POS_SCALE"
  reward.scales.joint_acc_l2=-2.5e-7
  reward.scales.joint_torque_l2=-1.0e-6
  "reward.scales.anti_shake_ang_vel=$ANTI_SHAKE_ANG_VEL_SCALE"
  "env.strict_foot_pos_threshold=$STRICT_FOOT_POS_THRESHOLD"
  "env.strict_anchor_ori_error_sq=$STRICT_ANCHOR_ORI_ERROR_SQ"
  "env.strict_height_threshold=$STRICT_HEIGHT_THRESHOLD"
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
  echo "Hydra compose passed (stage=$CURRICULUM_STAGE, motion_entries=${#NPZ_FILES[@]}, focus=$FOCUS_PATTERN, focus_repeat=$FOCUS_REPEAT, lr=$LEARNING_RATE, adaptive_lr_max=$ADAPTIVE_LR_MAX, entropy=$ENTROPY_COEF, root_pos_scale=$ROOT_POS_SCALE, action_rate=$ACTION_RATE_L2_SCALE, target_rate=$TARGET_RATE_L2_SCALE, upper_body_joint_pos=$UPPER_BODY_JOINT_POS_SCALE, anti_shake=$ANTI_SHAKE_ANG_VEL_SCALE, strict_foot=$STRICT_FOOT_POS_THRESHOLD, strict_ori_sq=$STRICT_ANCHOR_ORI_ERROR_SQ, strict_height=$STRICT_HEIGHT_THRESHOLD, init_std=$INIT_STD)"
  exit 0
fi

LOG="$WORK_DIR/full_train_gr2_stage${CURRICULUM_STAGE}_$(date +%Y%m%d_%H%M%S).log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
setsid nohup uv run --no-sync python scripts/train_rsl_rl.py "${TRAIN_ARGS[@]}" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched GR2 stage $CURRICULUM_STAGE (${#NPZ_FILES[@]} motion entries, focus=$FOCUS_PATTERN, focus_repeat=$FOCUS_REPEAT, lr=$LEARNING_RATE, adaptive_lr_max=$ADAPTIVE_LR_MAX, entropy=$ENTROPY_COEF, root_pos_scale=$ROOT_POS_SCALE, action_rate=$ACTION_RATE_L2_SCALE, target_rate=$TARGET_RATE_L2_SCALE, upper_body_joint_pos=$UPPER_BODY_JOINT_POS_SCALE, anti_shake=$ANTI_SHAKE_ANG_VEL_SCALE, strict_foot=$STRICT_FOOT_POS_THRESHOLD, strict_ori_sq=$STRICT_ANCHOR_ORI_ERROR_SQ, strict_height=$STRICT_HEIGHT_THRESHOLD, init_std=$INIT_STD), log=$LOG"