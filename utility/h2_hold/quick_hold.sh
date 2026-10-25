#!/usr/bin/env bash
# Quick H2 hold test: a few clips in parallel on CPU, then a comparison / report.
#
# usage: utility/h2_hold/quick_hold.sh <ckpt> <outdir> [<baseline_dir>]
#   with <baseline_dir>: compare.py <baseline_dir> <outdir>; otherwise analyze.py <outdir>.
# env:
#   CLIP_DIR  dir with *_hold.npz + manifest.json (make_hold_clips.py)  [/tmp/h2hold/clips]
#   CLIPS     space-separated clip stems (without _hold.npz)
#             [stand victory_dance_macarena_R_001__A321_h2 washing_dishes_R_002__A298_h2
#              walk_forward_loop_001__A021_h2]
#   JOBS      parallel rollouts (each CPU-only, OMP_NUM_THREADS=2)  [4]
#   EXTRA     extra run_hold.py args, e.g. "--scene <dir>/scene_flat.xml"
set -euo pipefail
if [ $# -lt 2 ]; then sed -n '2,13p' "$0"; exit 1; fi
CK=$(realpath "$1"); OUT=$(realpath -m "$2"); BASE=${3:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
CLIP_DIR=$(realpath "${CLIP_DIR:-/tmp/h2hold/clips}")
CLIPS=${CLIPS:-"stand victory_dance_macarena_R_001__A321_h2 washing_dishes_R_002__A298_h2 walk_forward_loop_001__A021_h2"}
JOBS=${JOBS:-4}; EXTRA=${EXTRA:-}
MANIFEST="$CLIP_DIR/manifest.json"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST (run make_hold_clips.py)"; exit 1; }
[ -f "$CK" ] || { echo "missing checkpoint $CK"; exit 1; }
mkdir -p "$OUT"; cd "$ROOT"

for b in $CLIPS; do
  if grep -q "\"${b}_hold.npz\"" "$MANIFEST"; then echo "$b"; else echo "skip $b (not in manifest)" >&2; fi
done | xargs -P "$JOBS" -I{} bash -c '
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 uv run --no-sync python "$0/run_hold.py" \
    "$1/{}_hold.npz" "$2/{}.npz" --ckpt "$3" $4 > "$2/{}.log" 2>&1 \
    && echo "done {}" || echo "FAILED {} (see $2/{}.log)"' "$HERE" "$CLIP_DIR" "$OUT" "$CK" "$EXTRA"

if [ -n "$BASE" ]; then
  uv run --no-sync python "$HERE/compare.py" "$BASE" "$OUT" "$(basename "$BASE")" "$(basename "$OUT")"
else
  uv run --no-sync python "$HERE/analyze.py" "$OUT"
fi
