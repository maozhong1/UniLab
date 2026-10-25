#!/usr/bin/env bash
# Full H2 hold suite: every clip in the manifest, optionally K perturbed seeds per clip.
#
# usage: utility/h2_hold/run_suite.sh <ckpt> <outdir> [K]
#   <outdir>/<clip>.npz             unperturbed rollout
#   <outdir>/seeds/<clip>__s<k>.npz perturbed rollouts, k = 1..K (K default 0)
#   then prints analyze.py <outdir> <outdir>/seeds
# env:
#   CLIP_DIR    dir with *_hold.npz + manifest.json  [/tmp/h2hold/clips]
#   JOBS        parallel rollouts (CPU-only, OMP_NUM_THREADS=2)  [4]
#   POSE_NOISE  run_hold.py --pose-noise for seeds  [0.02]
#   VEL_NOISE   run_hold.py --vel-noise for seeds   [0.05]
#   EXTRA       extra run_hold.py args for all runs, e.g. "--scene <dir>/scene_flat.xml"
set -euo pipefail
if [ $# -lt 2 ]; then sed -n '2,14p' "$0"; exit 1; fi
CK=$(realpath "$1"); OUT=$(realpath -m "$2"); K=${3:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
CLIP_DIR=$(realpath "${CLIP_DIR:-/tmp/h2hold/clips}")
JOBS=${JOBS:-4}; POSE_NOISE=${POSE_NOISE:-0.02}; VEL_NOISE=${VEL_NOISE:-0.05}; EXTRA=${EXTRA:-}
MANIFEST="$CLIP_DIR/manifest.json"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST (run make_hold_clips.py)"; exit 1; }
[ -f "$CK" ] || { echo "missing checkpoint $CK"; exit 1; }
mkdir -p "$OUT"; if [ "$K" -gt 0 ]; then mkdir -p "$OUT/seeds"; fi
cd "$ROOT"
export HERE CLIP_DIR CK POSE_NOISE VEL_NOISE EXTRA

# job lines: <clip stem> <out npz path> <seed>   (seed 0 = unperturbed)
uv run --no-sync python - "$MANIFEST" "$OUT" "$K" <<'PY' |
import json, sys
manifest, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
for name in json.load(open(manifest)):
    stem = name[: -len("_hold.npz")]
    print(stem, f"{out}/{stem}.npz", 0)
    for s in range(1, k + 1):
        print(stem, f"{out}/seeds/{stem}__s{s}.npz", s)
PY
  xargs -P "$JOBS" -n 3 bash -c '
  stem=$0; o=$1; seed=$2
  noise=""; [ "$seed" -gt 0 ] && noise="--seed $seed --pose-noise $POSE_NOISE --vel-noise $VEL_NOISE"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 uv run --no-sync python "$HERE/run_hold.py" \
    "$CLIP_DIR/${stem}_hold.npz" "$o" --ckpt "$CK" $noise $EXTRA > "${o%.npz}.log" 2>&1 \
    && echo "done $(basename "$o")" || echo "FAILED $(basename "$o") (see ${o%.npz}.log)"'

if [ "$K" -gt 0 ]; then
  uv run --no-sync python "$HERE/analyze.py" "$OUT" "$OUT/seeds"
else
  uv run --no-sync python "$HERE/analyze.py" "$OUT"
fi
