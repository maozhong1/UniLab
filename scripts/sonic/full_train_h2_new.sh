#!/usr/bin/env bash
set -euo pipefail

# H2 SONIC from-scratch curriculum: 1 (base clips) -> 2.5 (+manipulation) -> 2.8 (harder clips).
# Standalone: does not call or share state with full_train_h2.sh.
#
#   * gains: robots/h2/h2.xml as is (official hips/knees/hip yaw 200/4, ankle roll 40/2,
#     waist/arm/wrist; ankle pitch 70/3.5, above the official 40/2).
#   * data: sole-centre retarget (npz_*_new from g1_csv_to_h2_npz2.py --align_sole_center
#     --foot_forward_extra 0.025). Every clip ending in double support gets a static hold of
#     its last frame (scripts/motion/append_hold_tail_h2.py), because deploy holds the final
#     frame while training would otherwise truncate at the clip end.
#   * rewards: joint_soft_limit + joint_target_limit from the start so the policy never learns
#     to lean on the rigid joint stops (right ankle roll / knee hyperextension).
#   * rewards: feet_slide (reference-stance gated) -1.0 in every stage. Without it the policy
#     shuffled its feet during a deploy-style hold; resumed at stage-1 iter 1400, it went
#     from ~7 lifts / 23 cm drift to 0 lifts / <2 cm on the 6 s hold suite by iter 1800.
#   * rewards otherwise follow the official SONIC H2 config (gear_sonic sonic_h2.yaml):
#     no feet_yaw, feet_acc (joint_acc_l2) at -2.5e-6.
# Gate every stage on the hold test (utility/h2_hold), not on training curves alone.
#
# Usage (stage 1 from scratch; later stages resume the previous run's newest checkpoint
# unless CHECKPOINT is set):
#   CURRICULUM_STAGE=1   bash scripts/sonic/full_train_h2_new.sh
#   CURRICULUM_STAGE=2.5 LOAD_RUN=<stage-1 run dir>   bash scripts/sonic/full_train_h2_new.sh
#   CURRICULUM_STAGE=2.8 LOAD_RUN=<stage-2.5 run dir> bash scripts/sonic/full_train_h2_new.sh
# Every variable below can be overridden from the environment. DRY_RUN=1 prepares the data
# and checks the Hydra compose without training.

UNILAB="${UNILAB:-/home/maozhong/work/my_sonic/UniLab}"
SOURCE_NPZ_DIR="${SOURCE_NPZ_DIR:-$HOME/work/sonic_vla_infer/bones_seed_3k_new/npz_h2_29dof_new}"
MANIPULATION_NPZ_DIR="${MANIPULATION_NPZ_DIR:-$HOME/work/sonic_vla_infer/bones-seed/selections/upright_nzp_h2_29dof_new}"
WORK_DIR="${WORK_DIR:-/tmp/sonic_train}"
CURRICULUM_STAGE="${CURRICULUM_STAGE:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_STEPS="${NUM_STEPS:-24}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
INIT_STD="${INIT_STD:-0.50}"
STRICT_FOOT_POS_THRESHOLD="${STRICT_FOOT_POS_THRESHOLD:-0.50}"
STRICT_ANCHOR_ORI_ERROR_SQ="${STRICT_ANCHOR_ORI_ERROR_SQ:-0.80}"
STRICT_HEIGHT_THRESHOLD="${STRICT_HEIGHT_THRESHOLD:-0.45}"
ANTI_SHAKE_ANG_VEL_SCALE="${ANTI_SHAKE_ANG_VEL_SCALE:-2.5e-3}"
ANTI_SHAKE_ANG_VEL_SCALE="-${ANTI_SHAKE_ANG_VEL_SCALE#-}"
# Official SONIC H2 feet_acc weight (G1 default -2.5e-7); ankle pitch here is stiffer than
# the official 40, so the ankle-acceleration penalty matters at least as much.
JOINT_ACC_L2_SCALE="${JOINT_ACC_L2_SCALE:--2.5e-6}"
# Not in official SONIC; stops the hold-phase foot shuffling (see the header).
FEET_SLIDE_SCALE="${FEET_SLIDE_SCALE:--1.0}"
FEET_YAW_SCALE="${FEET_YAW_SCALE:-0.0}"

# Per-stage defaults. MAX_ITERATIONS is added on top of the resumed iteration.
case "$CURRICULUM_STAGE" in
  1)
    # Calm filter on the base 3k set only; manipulation clips join at 2.5. Entropy 0.004
    # explores more from scratch.
    FILTER_ARGS=(--body-ang-p99 8.0 --joint-vel-p99 7.0 --base-lin-p99 1.8 --base-ang-p99 2.5
                 --base-z-range 0.25 --foot-height-max 0.45)
    MERGE_MANIPULATION=0
    : "${MAX_ITERATIONS:=3500}"
    : "${LEARNING_RATE:=2e-5}" "${ADAPTIVE_LR_MAX:=2e-4}" "${ENTROPY_COEF:=0.004}"
    : "${ROOT_POS_SCALE:=0.5}" "${ACTION_RATE_L2_SCALE:=-0.01}" "${TARGET_RATE_L2_SCALE:=0.0}"
    : "${UPPER_BODY_JOINT_POS_SCALE:=0.0}" "${STD_UPPER_BODY_JOINT_POS:=0.5}"
    # No stop-leaning habit exists yet, so smaller weights than later stages suffice; a
    # random init (action std 0.5) keeps PD targets inside the ranges (~0 penalty).
    : "${JOINT_SOFT_LIMIT_SCALE:=-2.0}" "${JOINT_TARGET_LIMIT_SCALE:=-0.5}"
    : "${HOLD_TAIL_S:=1.5}"
    : "${CRITIC_WARMUP:=0}"
    : "${FOCUS_PATTERN:=}" "${FOCUS_REPEAT:=1}" "${EXTRA_FOCUS:=}"
    ;;
  2.5)
    FILTER_ARGS=(--body-ang-p99 9.0 --joint-vel-p99 7.5 --base-lin-p99 2.0 --base-ang-p99 3.0
                 --base-z-range 0.30 --foot-height-max 0.50)
    MERGE_MANIPULATION=1
    : "${MAX_ITERATIONS:=2700}"
    : "${LEARNING_RATE:=1e-5}" "${ADAPTIVE_LR_MAX:=2e-4}" "${ENTROPY_COEF:=0.003}"
    : "${ROOT_POS_SCALE:=0.75}" "${ACTION_RATE_L2_SCALE:=0.0}" "${TARGET_RATE_L2_SCALE:=-0.05}"
    # std 0.5: 0.2 saturates the upper-body reward on manipulation clips (arm error 0.3-0.5 rad).
    : "${UPPER_BODY_JOINT_POS_SCALE:=0.5}" "${STD_UPPER_BODY_JOINT_POS:=0.5}"
    : "${JOINT_SOFT_LIMIT_SCALE:=-3.0}" "${JOINT_TARGET_LIMIT_SCALE:=-1.0}"
    : "${HOLD_TAIL_S:=2.0}"
    : "${CRITIC_WARMUP:=0}"
    : "${FOCUS_PATTERN:=*macarena*.npz}" "${FOCUS_REPEAT:=5}" "${EXTRA_FOCUS:=}"
    ;;
  2.8)
    FILTER_ARGS=(--body-ang-p99 10.0 --joint-vel-p99 8.5 --base-lin-p99 2.0 --base-ang-p99 3.5
                 --base-z-range 0.30 --foot-height-max 0.55)
    MERGE_MANIPULATION=1
    : "${MAX_ITERATIONS:=2000}"
    : "${LEARNING_RATE:=1e-5}" "${ADAPTIVE_LR_MAX:=1e-4}" "${ENTROPY_COEF:=0.002}"
    : "${ROOT_POS_SCALE:=1.0}" "${ACTION_RATE_L2_SCALE:=0.0}" "${TARGET_RATE_L2_SCALE:=-0.02}"
    : "${UPPER_BODY_JOINT_POS_SCALE:=1.5}" "${STD_UPPER_BODY_JOINT_POS:=0.3}"
    : "${JOINT_SOFT_LIMIT_SCALE:=-3.0}" "${JOINT_TARGET_LIMIT_SCALE:=-1.0}"
    : "${HOLD_TAIL_S:=2.0}"
    # ~250 harder clips join: short critic burn-in.
    : "${CRITIC_WARMUP:=50}"
    : "${FOCUS_PATTERN:=*macarena*.npz}" "${FOCUS_REPEAT:=5}"
    # Rare stop / backward / manipulation clips get extra weight.
    : "${EXTRA_FOCUS:=walk_backward_stop_*:4 walk_backward_loop_*:2 walk_forward_stop_*:2 *peeling*:3 *washing*:3 *lighter*:3 *operating*:2}"
    ;;
  *)
    echo "ERROR: CURRICULUM_STAGE must be 1, 2.5 or 2.8" >&2
    exit 2
    ;;
esac

if [[ "$CURRICULUM_STAGE" != "1" && -z "${LOAD_RUN:-}" ]]; then
  echo "ERROR: LOAD_RUN is required for stage $CURRICULUM_STAGE" >&2
  exit 2
fi
if [[ ! "$FOCUS_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: FOCUS_REPEAT must be a positive integer" >&2
  exit 2
fi
read -ra EXTRA_FOCUS_SPECS <<< "$EXTRA_FOCUS"
for spec in "${EXTRA_FOCUS_SPECS[@]}"; do
  if [[ "$spec" != *:* || ! "${spec##*:}" =~ ^[1-9][0-9]*$ || -z "${spec%:*}" ]]; then
    echo "ERROR: EXTRA_FOCUS entries must be glob:repeat with a positive repeat: $spec" >&2
    exit 2
  fi
done

cd "$UNILAB"
mkdir -p "$WORK_DIR/motion_override"
TAG="h2new_stage${CURRICULUM_STAGE//./_}"
SRC_DIR="$WORK_DIR/${TAG}_source"
FILTERED_DIR="$WORK_DIR/${TAG}"
NPZ_DIR="$WORK_DIR/${TAG}_hold"

# 1) Source set: base clips (+ manipulation clips from 2.5), symlinked into one directory.
mkdir -p "$SRC_DIR"
find "$SRC_DIR" -maxdepth 1 -name '*.npz' -type l -delete
if find "$SRC_DIR" -maxdepth 1 -name '*.npz' ! -type l | grep -q .; then
  echo "ERROR: refusing to touch real files in $SRC_DIR" >&2
  exit 1
fi
SOURCES=("$SOURCE_NPZ_DIR")
if [[ "$MERGE_MANIPULATION" == "1" ]]; then
  SOURCES+=("$MANIPULATION_NPZ_DIR")
fi
for dir in "${SOURCES[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "ERROR: motion source directory does not exist: $dir" >&2
    exit 1
  fi
  for src in "$dir"/*.npz; do
    target="$SRC_DIR/${src##*/}"
    if [[ -e "$target" ]]; then
      echo "ERROR: duplicate clip name: ${src##*/}" >&2
      exit 1
    fi
    ln -s "$src" "$target"
  done
done

# 2) Stage filter (symlinks), 3) hold-extended copies of clips ending in double support
#    (written under WORK_DIR; the source npz files are never modified).
uv run --no-sync python scripts/motion/filter_calm_h2.py \
  --src "$SRC_DIR" --dst "$FILTERED_DIR" --show-dropped 0 "${FILTER_ARGS[@]}"
if [[ "$HOLD_TAIL_S" == "0" ]]; then
  NPZ_DIR="$FILTERED_DIR"
else
  uv run --no-sync python scripts/motion/append_hold_tail_h2.py \
    --src "$FILTERED_DIR" --dst "$NPZ_DIR" --hold-s "$HOLD_TAIL_S"
fi

shopt -s nullglob
NPZ_FILES=("$NPZ_DIR"/*.npz)
shopt -u nullglob
if [[ ${#NPZ_FILES[@]} -eq 0 ]]; then
  echo "ERROR: no NPZ clips in $NPZ_DIR" >&2
  exit 1
fi
UNIQUE_NPZ_FILES=("${NPZ_FILES[@]}")

# 4) Oversampling by repeating motion_file entries.
oversample() {  # <glob> <repeat>
  local pattern="$1" repeat="$2" matched=() path r
  for path in "${UNIQUE_NPZ_FILES[@]}"; do
    [[ "${path##*/}" == $pattern ]] && matched+=("$path")
  done
  if [[ ${#matched[@]} -eq 0 ]]; then
    echo "WARNING: pattern matched no clips in this stage, skipping: $pattern" >&2
    return
  fi
  for ((r = 1; r < repeat; r++)); do
    NPZ_FILES+=("${matched[@]}")
  done
  echo "oversampled ${#matched[@]} clips ${repeat}x ($pattern)"
}
if [[ -n "$FOCUS_PATTERN" && "$FOCUS_REPEAT" -gt 1 ]]; then
  oversample "$FOCUS_PATTERN" "$FOCUS_REPEAT"
fi
for spec in "${EXTRA_FOCUS_SPECS[@]}"; do
  oversample "${spec%:*}" "${spec##*:}"
done

# 5) Motion list as a temporary Hydra config (keeps thousands of paths out of argv).
MOTION_OVERRIDE_NAME="${TAG}_$$"
{
  echo "# @package _global_"
  echo "env:"
  echo "  motion_file:"
  printf '    - %s\n' "${NPZ_FILES[@]}"
} > "$WORK_DIR/motion_override/$MOTION_OVERRIDE_NAME.yaml"
echo "wrote ${#NPZ_FILES[@]} motion entries -> $WORK_DIR/motion_override/$MOTION_OVERRIDE_NAME.yaml"

# 6) Resume (stages after 1, or a restarted stage 1 when LOAD_RUN is set).
RESUME_ARGS=(algo.resume=false)
if [[ -n "${LOAD_RUN:-}" ]]; then
  RUN_DIR="$UNILAB/logs/rsl_rl_ppo/H2SonicMotionTracking/$LOAD_RUN"
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: H2 run directory does not exist: $RUN_DIR" >&2
    exit 1
  fi
  if [[ -n "${CHECKPOINT:-}" ]]; then
    CKPT="${CHECKPOINT#model_}"
    CKPT="${CKPT%.pt}"
  else
    CKPT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
      | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
  fi
  if [[ ! "${CKPT:-}" =~ ^[0-9]+$ || ! -f "$RUN_DIR/model_$CKPT.pt" ]]; then
    echo "ERROR: checkpoint not found in $RUN_DIR (CHECKPOINT=${CHECKPOINT:-newest})" >&2
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
  "reward.scales.joint_acc_l2=$JOINT_ACC_L2_SCALE"
  reward.scales.joint_torque_l2=-1.0e-6
  "reward.scales.anti_shake_ang_vel=$ANTI_SHAKE_ANG_VEL_SCALE"
  "reward.scales.feet_slide=$FEET_SLIDE_SCALE"
  "reward.scales.feet_yaw=$FEET_YAW_SCALE"
  "reward.scales.joint_soft_limit=$JOINT_SOFT_LIMIT_SCALE"
  "reward.scales.joint_target_limit=$JOINT_TARGET_LIMIT_SCALE"
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
SUMMARY="stage=$CURRICULUM_STAGE entries=${#NPZ_FILES[@]} hold_tail_s=$HOLD_TAIL_S iters=+$MAX_ITERATIONS \
lr=$LEARNING_RATE lr_max=$ADAPTIVE_LR_MAX entropy=$ENTROPY_COEF root_pos=$ROOT_POS_SCALE \
action_rate=$ACTION_RATE_L2_SCALE target_rate=$TARGET_RATE_L2_SCALE upper=$UPPER_BODY_JOINT_POS_SCALE/std$STD_UPPER_BODY_JOINT_POS \
feet_slide=$FEET_SLIDE_SCALE feet_yaw=$FEET_YAW_SCALE joint_acc=$JOINT_ACC_L2_SCALE soft_limit=$JOINT_SOFT_LIMIT_SCALE \
target_limit=$JOINT_TARGET_LIMIT_SCALE critic_warmup=$CRITIC_WARMUP focus=[$FOCUS_PATTERN x$FOCUS_REPEAT] extra=[$EXTRA_FOCUS]"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  uv run --no-sync python scripts/train_rsl_rl.py "${TRAIN_ARGS[@]}" --cfg job >/dev/null
  echo "Hydra compose passed ($SUMMARY)"
  exit 0
fi

LOG="$WORK_DIR/full_train_${TAG}_$(date +%Y%m%d_%H%M%S).log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
setsid nohup uv run --no-sync python scripts/train_rsl_rl.py "${TRAIN_ARGS[@]}" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched H2 from-scratch $SUMMARY"
echo "log=$LOG"

# Validation (per stage, before moving on):
#   * training: termination ~0, episode length near the clip length, reward/joint_soft_limit
#     and reward/joint_target_limit near 0, diagnostic/estimated_torque_saturation_ratio well
#     below 1/29 (~3.4% = one joint pinned at its limit).
#   * hold test (utility/h2_hold): npz_new clips with the last frame held 6 s, several
#     perturbed rollouts per clip: no steps, |pelvis pitch error| <= 3 deg, CoP mid-foot,
#     no joint resting on its stop. Then deploy with gains matching robots/h2/h2.xml.
