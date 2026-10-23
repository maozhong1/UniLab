#!/usr/bin/env bash
set -euo pipefail

# H2 (29-DOF, head welded like Unitree G1) SONIC curriculum training.
# Stage 1 starts from scratch. Later stages require LOAD_RUN and resume its newest
# checkpoint unless CHECKPOINT is set. Stages 2.5, 2.6 and 2.8 merge manipulation clips
# into the base dataset before filtering. Set NPZ_DIR to bypass filtering.
# Stage 2.6 = stage 2.5 data under the retuned ankle (Kp 40) / wrist pitch-yaw (Kp 20) PD
# and 4-sphere sole contact (elliptic cone),
# plus reference-stance feet_slide/feet_yaw. Resume it from a checkpoint converted with
# scripts/sonic/rescale_h2_action_ckpt.py (the action scale changed for those joints).
#   task:   h2_motion_tracking/sonic_full_train (H2SonicMotionTracking env)
#   actor:  SonicH2ActorModel, obs 1570 = enc(640)+proprio(930), action 29
#   critic: 1645 privileged_mf_hist (+env.critic_privileged_mf_hist=true), fresh
#   data:   29-DOF H2 npz from scripts/motion/g1_csv_to_h2_npz2.py (regenerate post head-weld)

UNILAB="${UNILAB:-/home/maozhong/work/my_sonic/UniLab}"
SOURCE_NPZ_DIR="${SOURCE_NPZ_DIR:-$HOME/work/sonic_vla_infer/bones_seed_3k_new/npz_h2_29dof}"
MANIPULATION_NPZ_DIR="${MANIPULATION_NPZ_DIR:-$HOME/work/sonic_vla_infer/bones-seed/selections/upright_nzp_h2_29dof}"
CURRICULUM_STAGE="${CURRICULUM_STAGE:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_STEPS="${NUM_STEPS:-24}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
WORK_DIR="${WORK_DIR:-/tmp/sonic_train}"

if [[ "$CURRICULUM_STAGE" == "2.8" ]]; then
  # The adaptive-KL schedule floors the LR at 1e-5, so a lower base LR is a no-op. Stage 2.5
  # oscillated 1e-5..2.25e-5; the 3e-5 ceiling only guards against a ramp when KL collapses
  # as action std shrinks. target_rate stays at the 2.6 value (arms are only ~22% of it;
  # legs/ankles/waist share the rest). Root position and upper-body joints are up-weighted
  # because both still track poorly after 2.6 (upper-body reward flat at ~0.41).
  DEFAULT_LEARNING_RATE=1e-5
  DEFAULT_ADAPTIVE_LR_MAX=3e-5
  DEFAULT_ENTROPY_COEF=0.002
  DEFAULT_ROOT_POS_SCALE=1.0
  DEFAULT_ACTION_RATE_L2_SCALE=0.0
  DEFAULT_TARGET_RATE_L2_SCALE=-0.02
  DEFAULT_UPPER_BODY_JOINT_POS_SCALE=1.0
  DEFAULT_ANTI_SHAKE_ANG_VEL_SCALE=-2.5e-3
  # Carry the stage 2.6 foot-slip terms and upper-body std forward (dropping them would
  # silently undo 2.6), and burn the critic in on the ~230 harder clips 2.8 adds.
  DEFAULT_FEET_SLIDE_SCALE=-1.0
  DEFAULT_FEET_YAW_SCALE=-0.1
  DEFAULT_STD_UPPER_BODY_JOINT_POS=0.5
  DEFAULT_CRITIC_WARMUP=50
  # 2.6 eval: backward stance slip did not improve (rare in the data: backward
  # ~1.3%, sideways ~15% of 2.8 clips), so give them extra weight.
  DEFAULT_EXTRA_FOCUS='walk_backward_*:2'
  DEFAULT_FOCUS_PATTERN='*macarena*.npz'
  DEFAULT_FOCUS_REPEAT=5
elif [[ "$CURRICULUM_STAGE" == "2.6" ]]; then
  # Same data and reward base as 2.5. New: foot-slip penalties, and a wider upper-body
  # joint std (0.2 rad saturated the reward to ~0 on manipulation/dance clips, whose
  # arm error is ~0.3-0.5 rad, leaving no gradient). Critic burn-in for the PD change.
  # Foot terms sized from the first 2.6 run: at -0.2 / -0.05 they logged -0.009 / -0.04
  # against ~4.2 of tracking reward (feet_slide 0.2%, below every smoothness term).
  DEFAULT_LEARNING_RATE=1e-5
  DEFAULT_ADAPTIVE_LR_MAX=3e-5
  DEFAULT_ENTROPY_COEF=0.003
  DEFAULT_ROOT_POS_SCALE=0.75
  DEFAULT_ACTION_RATE_L2_SCALE=0.0
  DEFAULT_TARGET_RATE_L2_SCALE=-0.05
  DEFAULT_UPPER_BODY_JOINT_POS_SCALE=0.5
  DEFAULT_ANTI_SHAKE_ANG_VEL_SCALE=-2.5e-3
  DEFAULT_FEET_SLIDE_SCALE=-1.0
  DEFAULT_FEET_YAW_SCALE=-0.1
  DEFAULT_STD_UPPER_BODY_JOINT_POS=0.5
  DEFAULT_CRITIC_WARMUP=50
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
  # The Stage 2.5 body-angular limit (9.0) admits some, but not necessarily all,
  # macarena clips; focus oversampling is skipped with a warning if none pass all filters.
  DEFAULT_FOCUS_PATTERN='*macarena*.npz'
  DEFAULT_FOCUS_REPEAT=5
else
  # Stages 1, 2, 3, full. Oversample macarena by default (the 3k base set has only a small
  # fraction). Stage 1/2 filters reject its clips; Stage 2.5 may admit some at body_ang_p99
  # <= 9, with later stages admitting more. Stage 1 overrides this to no focus (see below).
  DEFAULT_LEARNING_RATE=2e-5
  DEFAULT_ADAPTIVE_LR_MAX=2e-4
  DEFAULT_ENTROPY_COEF=0.004
  DEFAULT_ROOT_POS_SCALE=0.5
  DEFAULT_ACTION_RATE_L2_SCALE=-0.01
  DEFAULT_TARGET_RATE_L2_SCALE=0.0
  DEFAULT_UPPER_BODY_JOINT_POS_SCALE=0.0
  DEFAULT_ANTI_SHAKE_ANG_VEL_SCALE=-2.5e-3
  DEFAULT_FOCUS_PATTERN='*macarena*.npz'
  DEFAULT_FOCUS_REPEAT=5
fi

DEFAULT_FEET_SLIDE_SCALE="${DEFAULT_FEET_SLIDE_SCALE:-0.0}"
DEFAULT_FEET_YAW_SCALE="${DEFAULT_FEET_YAW_SCALE:-0.0}"
DEFAULT_STD_UPPER_BODY_JOINT_POS="${DEFAULT_STD_UPPER_BODY_JOINT_POS:-0.2}"
DEFAULT_CRITIC_WARMUP="${DEFAULT_CRITIC_WARMUP:-0}"
DEFAULT_EXTRA_FOCUS="${DEFAULT_EXTRA_FOCUS:-}"

# Stage 1 is the calm from-scratch stage; its filter drops every macarena clip, so disable
# focus here to avoid a spurious "matched no clips" warning (stage 2/3/full keep it).
if [[ "$CURRICULUM_STAGE" == "1" ]]; then
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
FEET_SLIDE_SCALE="${FEET_SLIDE_SCALE:-$DEFAULT_FEET_SLIDE_SCALE}"
FEET_YAW_SCALE="${FEET_YAW_SCALE:-$DEFAULT_FEET_YAW_SCALE}"
STD_UPPER_BODY_JOINT_POS="${STD_UPPER_BODY_JOINT_POS:-$DEFAULT_STD_UPPER_BODY_JOINT_POS}"
FOCUS_PATTERN="${FOCUS_PATTERN:-$DEFAULT_FOCUS_PATTERN}"
FOCUS_REPEAT="${FOCUS_REPEAT:-$DEFAULT_FOCUS_REPEAT}"
# Extra per-pattern oversampling on top of FOCUS_PATTERN: space-separated "glob:repeat"
# entries matched against the stage's unique clips, e.g. 'walk_backward_*:2'. Set to a
# single space to disable a stage default.
EXTRA_FOCUS="${EXTRA_FOCUS:-$DEFAULT_EXTRA_FOCUS}"
read -ra EXTRA_FOCUS_SPECS <<< "$EXTRA_FOCUS"
for spec in "${EXTRA_FOCUS_SPECS[@]}"; do
  if [[ "$spec" != *:* || ! "${spec##*:}" =~ ^[1-9][0-9]*$ || -z "${spec%:*}" ]]; then
    echo "ERROR: EXTRA_FOCUS entries must be glob:repeat with a positive repeat: $spec" >&2
    exit 2
  fi
done
INIT_STD="${INIT_STD:-0.50}"
# Resume-only: freeze the actor for N iterations so the critic re-fits first (e.g. after the
# motion data or PD gains change under a warm policy). 0 = off.
CRITIC_WARMUP="${CRITIC_WARMUP:-$DEFAULT_CRITIC_WARMUP}"

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
    2)
      FILTER_ARGS=(
        --body-ang-p99 8.0 --joint-vel-p99 7.0
        --base-lin-p99 1.8 --base-ang-p99 2.5
        --base-z-range 0.25 --foot-height-max 0.45
      )
      ;;
    2.5|2.6)
      FILTER_ARGS=(
        --body-ang-p99 9.0 --joint-vel-p99 7.5
        --base-lin-p99 2.0 --base-ang-p99 3.0
        --base-z-range 0.30 --foot-height-max 0.50
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
      echo "ERROR: CURRICULUM_STAGE must be 1, 2, 2.5, 2.6, 2.8, 3, or full" >&2
      exit 2
      ;;
  esac

  if [[ "$CURRICULUM_STAGE" == "full" ]]; then
    NPZ_DIR="$SOURCE_NPZ_DIR"
  else
    FILTER_SOURCE_DIR="$SOURCE_NPZ_DIR"
    STAGE_DIR_SUFFIX="${CURRICULUM_STAGE//./_}"
    if [[ "$CURRICULUM_STAGE" == "2.5" || "$CURRICULUM_STAGE" == "2.6" || "$CURRICULUM_STAGE" == "2.8" ]]; then
      FILTER_SOURCE_DIR="$WORK_DIR/h2_stage${STAGE_DIR_SUFFIX}_source"
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
    NPZ_DIR="$WORK_DIR/h2_stage${STAGE_DIR_SUFFIX}"
    uv run --no-sync python scripts/motion/filter_calm_h2.py \
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

UNIQUE_NPZ_FILES=("${NPZ_FILES[@]}")

if [[ -n "$FOCUS_PATTERN" && "$FOCUS_REPEAT" -gt 1 ]]; then
  FOCUS_FILES=()
  for path in "${NPZ_FILES[@]}"; do
    if [[ "${path##*/}" == $FOCUS_PATTERN ]]; then
      FOCUS_FILES+=("$path")
    fi
  done
  if [[ ${#FOCUS_FILES[@]} -eq 0 ]]; then
    # Non-fatal: the stage filter may legitimately drop every focus clip (e.g. macarena
    # removed by the Stage 1 calm filter). Warn and train without oversampling.
    echo "WARNING: FOCUS_PATTERN matched no clips in this stage, skipping oversampling: $FOCUS_PATTERN" >&2
  else
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
fi

for spec in "${EXTRA_FOCUS_SPECS[@]}"; do
  pattern="${spec%:*}"
  extra_repeat="${spec##*:}"
  EXTRA_FILES=()
  for path in "${UNIQUE_NPZ_FILES[@]}"; do
    if [[ "${path##*/}" == $pattern ]]; then
      EXTRA_FILES+=("$path")
    fi
  done
  if [[ ${#EXTRA_FILES[@]} -eq 0 ]]; then
    echo "WARNING: EXTRA_FOCUS pattern matched no clips in this stage, skipping: $pattern" >&2
    continue
  fi
  for ((repeat = 1; repeat < extra_repeat; repeat++)); do
    NPZ_FILES+=("${EXTRA_FILES[@]}")
  done
  echo "oversampled ${#EXTRA_FILES[@]} extra-focus clips ${extra_repeat}x ($pattern)"
done

# Keep the 3000+ paths out of argv (Linux MAX_ARG_STRLEN) by composing a
# temporary Hydra config through hydra.searchpath.
MOTION_OVERRIDE_NAME="h2_stage_${CURRICULUM_STAGE//./_}_$$"
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
# Stages > 1 always resume a prior run (LOAD_RUN required). Stage 1 is from
# scratch unless LOAD_RUN is set, in which case it resumes (e.g. restart after
# a disk-full interruption).
if [[ "$CURRICULUM_STAGE" != "1" && -z "${LOAD_RUN:-}" ]]; then
  echo "ERROR: LOAD_RUN is required for curriculum stage $CURRICULUM_STAGE" >&2
  exit 2
fi
if [[ -n "${LOAD_RUN:-}" ]]; then
  RUN_DIR="$UNILAB/logs/rsl_rl_ppo/H2SonicMotionTracking/$LOAD_RUN"
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: H2 run directory does not exist: $RUN_DIR" >&2
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
fi

TRAIN_ARGS=(
  task=h2_motion_tracking/sonic_full_train
  training.device=xpu
  training.no_play=true
  "algo.actor.distribution_cfg.init_std=$INIT_STD"
  "${RESUME_ARGS[@]}"
  "algo.algorithm.learning_rate=$LEARNING_RATE"
  "algo.algorithm.entropy_coef=$ENTROPY_COEF"
  algo.algorithm.desired_kl=0.01
  "algo.algorithm.critic_warmup_iters=$CRITIC_WARMUP"
  "algo.algorithm.adaptive_lr_max=$ADAPTIVE_LR_MAX"
  "algo.num_steps_per_env=$NUM_STEPS"
  "reward.scales.motion_global_root_pos=$ROOT_POS_SCALE"
  "reward.scales.action_rate_l2=$ACTION_RATE_L2_SCALE"
  "reward.scales.target_rate_l2=$TARGET_RATE_L2_SCALE"
  "reward.scales.upper_body_joint_pos=$UPPER_BODY_JOINT_POS_SCALE"
  reward.scales.joint_acc_l2=-2.5e-7
  reward.scales.joint_torque_l2=-1.0e-6
  "reward.scales.anti_shake_ang_vel=$ANTI_SHAKE_ANG_VEL_SCALE"
  "reward.scales.feet_slide=$FEET_SLIDE_SCALE"
  "reward.scales.feet_yaw=$FEET_YAW_SCALE"
  "reward.std_upper_body_joint_pos=$STD_UPPER_BODY_JOINT_POS"
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
  echo "Hydra compose passed (stage=$CURRICULUM_STAGE, motion_entries=${#NPZ_FILES[@]}, focus=$FOCUS_PATTERN, focus_repeat=$FOCUS_REPEAT, extra_focus=[$EXTRA_FOCUS], lr=$LEARNING_RATE, adaptive_lr_max=$ADAPTIVE_LR_MAX, entropy=$ENTROPY_COEF, root_pos_scale=$ROOT_POS_SCALE, action_rate=$ACTION_RATE_L2_SCALE, target_rate=$TARGET_RATE_L2_SCALE, upper_body_joint_pos=$UPPER_BODY_JOINT_POS_SCALE, anti_shake=$ANTI_SHAKE_ANG_VEL_SCALE, feet_slide=$FEET_SLIDE_SCALE, feet_yaw=$FEET_YAW_SCALE, std_upper=$STD_UPPER_BODY_JOINT_POS, critic_warmup=$CRITIC_WARMUP, strict_foot=$STRICT_FOOT_POS_THRESHOLD, strict_ori_sq=$STRICT_ANCHOR_ORI_ERROR_SQ, strict_height=$STRICT_HEIGHT_THRESHOLD, init_std=$INIT_STD)"
  exit 0
fi

LOG="$WORK_DIR/full_train_h2_stage${CURRICULUM_STAGE}_$(date +%Y%m%d_%H%M%S).log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
setsid nohup uv run --no-sync python scripts/train_rsl_rl.py "${TRAIN_ARGS[@]}" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched H2 stage $CURRICULUM_STAGE (${#NPZ_FILES[@]} motion entries, focus=$FOCUS_PATTERN, focus_repeat=$FOCUS_REPEAT, extra_focus=[$EXTRA_FOCUS], lr=$LEARNING_RATE, adaptive_lr_max=$ADAPTIVE_LR_MAX, entropy=$ENTROPY_COEF, root_pos_scale=$ROOT_POS_SCALE, action_rate=$ACTION_RATE_L2_SCALE, target_rate=$TARGET_RATE_L2_SCALE, upper_body_joint_pos=$UPPER_BODY_JOINT_POS_SCALE, anti_shake=$ANTI_SHAKE_ANG_VEL_SCALE, feet_slide=$FEET_SLIDE_SCALE, feet_yaw=$FEET_YAW_SCALE, std_upper=$STD_UPPER_BODY_JOINT_POS, critic_warmup=$CRITIC_WARMUP, strict_foot=$STRICT_FOOT_POS_THRESHOLD, strict_ori_sq=$STRICT_ANCHOR_ORI_ERROR_SQ, strict_height=$STRICT_HEIGHT_THRESHOLD, init_std=$INIT_STD), log=$LOG"
