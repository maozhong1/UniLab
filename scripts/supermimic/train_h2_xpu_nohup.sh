#!/usr/bin/env bash
# Launch H2 SuperMimic training on the Intel XPU detached from the terminal (survives SSH /
# VSCode disconnects). Extra Hydra overrides are passed through, e.g.
#   scripts/supermimic/train_h2_xpu_nohup.sh algo.supermimic.schedule.total_iterations=10000
#   scripts/supermimic/train_h2_xpu_nohup.sh algo.resume=logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt
# A task=... argument replaces the default task, e.g.
#   scripts/supermimic/train_h2_xpu_nohup.sh task=h2_supermimic/mujoco_xpu_finetune algo.resume=<ckpt>
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/supermimic
LOG="logs/supermimic/train_h2_xpu_$(date +%Y%m%d_%H%M%S).log"
TASK=(task=h2_supermimic/mujoco_xpu)
for a in "$@"; do [[ $a == task=* ]] && TASK=(); done
PYTHONUNBUFFERED=1 setsid nohup uv run --no-sync python scripts/train_supermimic.py \
    "${TASK[@]}" "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > logs/supermimic/train_h2_xpu.pid
echo "started PID $(cat logs/supermimic/train_h2_xpu.pid), log: $LOG"
echo "follow:  tail -f $LOG | grep -v RuntimeWarning"
