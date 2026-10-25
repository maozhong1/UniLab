#!/usr/bin/env bash
# Block until <run_dir>/model_<iter>.pt exists, or the training process is gone.
#
# usage: utility/h2_hold/wait_ckpt.sh <run_dir> <iter> <process_pattern>
#   <process_pattern> is matched with `pgrep -f` (read-only), e.g. "train_rsl_rl.py.*h2_stage_2_9".
# Prints "ckpt <path>" (exit 0) or "training exited" (exit 2). Poll interval: POLL_S [60].
set -euo pipefail
if [ $# -ne 3 ]; then sed -n '2,6p' "$0"; exit 1; fi
RUN=$1; IT=$2; PAT=$3; POLL_S=${POLL_S:-60}
[ -d "$RUN" ] || { echo "no such run dir: $RUN"; exit 1; }
CK="$RUN/model_$IT.pt"
while true; do
  if [ -f "$CK" ]; then sleep 20; echo "ckpt $CK"; exit 0; fi  # let torch.save finish
  # ignore this script (and wrappers invoking it): their argv contains the pattern too
  if ! pgrep -af -- "$PAT" | grep -v -e wait_ckpt.sh -e "^$$ " | grep -q .; then
    if [ -f "$CK" ]; then echo "ckpt $CK"; exit 0; fi
    echo "training exited"; exit 2
  fi
  sleep "$POLL_S"
done
