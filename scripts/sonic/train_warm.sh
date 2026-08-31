#!/usr/bin/env bash
# Warm-start finetune of the sonic G1 encoder+decoder from sonic_release/last.pt.
#
# Byte-compat prerequisites (all verified via scripts/sonic/parity_harness.py):
#   - obs side : mujoco_to_isaaclab_perm=True (default in tracking_sonic.py)
#   - action   : +env.action_output_isaaclab_to_mujoco=true (set in sonic.yaml)
#   - no obs normalization, init_std=0.05, small LR/entropy (warm-start).
#
# Usage:
#   scripts/sonic/train_warm.sh smoke      # 16 envs, 2 iters — sanity only
#   scripts/sonic/train_warm.sh run        # 512 envs, 1000 iters, save every 200
# Env overrides: NPZ_DIR, CKPT, DEVICE, NUM_ENVS, ITERS, SAVE, FREEZE_ENCODER.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> UniLab root

MODE="${1:-smoke}"
NPZ_DIR="${NPZ_DIR:-/home/maozhong/work/sonic_vla_infer/bones_seed_subset/npz}"
# Default to ./last.pt at the repo root (the script cd's there above). Override with
# CKPT=/abs/path/to/sonic_release/last.pt.
CKPT="${CKPT:-./last.pt}"
DEVICE="${DEVICE:-xpu}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"   # true = decoder-only (VLA-safe token space)
LR="${LR:-0.0001}"                          # PPO learning rate (lower for gentler warm-start finetune)
ENCODER_LR="${ENCODER_LR:-}"                 # optional absolute encoder LR; when set (and encoder NOT frozen),
                                            # encoder uses this LR while decoder/critic use LR. Unset -> encoder uses LR.
CRITIC_LR="${CRITIC_LR:-}"                   # optional absolute critic LR (cold-critic cure). When set, critic uses
                                            # this LR while actor(decoder/encoder) uses LR. Official: LR=2e-5 CRITIC_LR=1e-3.

if [[ "$MODE" == "smoke" ]]; then
  NUM_ENVS="${NUM_ENVS:-16}"; ITERS="${ITERS:-2}"; SAVE="${SAVE:-1000}"
else
  NUM_ENVS="${NUM_ENVS:-512}"; ITERS="${ITERS:-1000}"; SAVE="${SAVE:-200}"
fi

# The UniLab venv reverts torch to +cu128 (pyproject pin); restore +xpu if needed.
if [[ "$DEVICE" == "xpu" ]]; then
  if ! uv run --no-sync python -c "import torch,sys; sys.exit(0 if torch.xpu.is_available() else 1)" 2>/dev/null; then
    echo "[train_warm] torch.xpu unavailable -> restoring +xpu wheel (cached)"
    uv pip install "torch==2.7.0+xpu" "pytorch-triton-xpu" \
      --index-url https://download.pytorch.org/whl/xpu
  fi
fi

[[ -f "$CKPT" ]] || { echo "[train_warm] ckpt not found: $CKPT" >&2; exit 1; }
LIST=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | paste -sd, -)
[[ -n "$LIST" ]] || { echo "[train_warm] no .npz in $NPZ_DIR" >&2; exit 1; }

# Optional per-group LRs. Encoder LR only meaningful when the encoder is trainable;
# critic LR always applies (critic is always trained).
ENC_LR_ARG=""
if [[ "$FREEZE_ENCODER" != "true" && -n "$ENCODER_LR" ]]; then
  ENC_LR_ARG="algo.algorithm.encoder_lr=$ENCODER_LR"
fi
CRIT_LR_ARG=""
if [[ -n "$CRITIC_LR" ]]; then
  CRIT_LR_ARG="algo.algorithm.critic_lr=$CRITIC_LR"
fi

echo "[train_warm] mode=$MODE device=$DEVICE envs=$NUM_ENVS iters=$ITERS save=$SAVE lr=$LR encoder_lr=${ENCODER_LR:-<=lr>} critic_lr=${CRITIC_LR:-<=lr>} freeze_encoder=$FREEZE_ENCODER"
echo "[train_warm] ckpt=$CKPT"
echo "[train_warm] motions=$(echo "$LIST" | tr ',' '\n' | wc -l) clips from $NPZ_DIR"

HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python scripts/train_rsl_rl.py \
  task=g1_motion_tracking/sonic training.device="$DEVICE" training.no_play=true \
  algo.actor.pretrained_ckpt="$CKPT" \
  algo.actor.freeze_encoder="$FREEZE_ENCODER" \
  algo.actor.distribution_cfg.init_std=0.05 \
  algo.algorithm.learning_rate="$LR" \
  $ENC_LR_ARG \
  $CRIT_LR_ARG \
  algo.algorithm.entropy_coef=0.001 \
  "+env.motion_file=[$LIST]" \
  algo.num_envs="$NUM_ENVS" algo.max_iterations="$ITERS" algo.save_interval="$SAVE"
