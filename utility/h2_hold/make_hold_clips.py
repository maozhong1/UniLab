"""Build "hold" clips: each source clip + N seconds of its last frame (velocities zeroed).

The policy is then asked to keep the final pose, which is what the deploy stack does when a
clip ends. A ``manifest.json`` ({clip_hold.npz: n_orig_frames}) is written next to the clips so
later scripts know where the hold phase starts without the source directory.

usage:
    uv run --no-sync python utility/h2_hold/make_hold_clips.py --dst /tmp/h2hold/clips \
        [--src <npz dir>] [--hold-s 6] \
        [--stand src/unilab/assets/motions/h2/stand.npz --stand-hold-s 3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_SRC = Path(
    "~/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/gear_sonic_deploy/h2/npz_new"
).expanduser()
ZERO_VEL_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")


def extend(src: Path, dst: Path, hold_s: float) -> int:
    """Append ``hold_s`` seconds of the last frame to ``src``; return the original frame count."""
    z = np.load(src)
    fps = int(np.asarray(z["fps"]).reshape(-1)[0])
    n_hold = int(round(hold_s * fps))
    out = {}
    for k in z.files:
        a = z[k]
        if k == "fps":
            out[k] = a
            continue
        rep = np.repeat(a[-1:], n_hold, axis=0)
        if k in ZERO_VEL_KEYS:
            rep = np.zeros_like(rep)
        out[k] = np.concatenate([a, rep], 0).astype(a.dtype)
    np.savez(dst, **out)
    n_orig = len(z["joint_pos"])
    print(f"{dst.name}: {n_orig} -> {len(out['joint_pos'])} frames")
    return n_orig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="dir of source *.npz clips")
    ap.add_argument("--dst", type=Path, required=True, help="output dir for *_hold.npz")
    ap.add_argument("--hold-s", type=float, default=6.0, help="hold seconds appended per clip")
    ap.add_argument("--stand", type=Path, default=None, help="optional stand npz (-> stand_hold)")
    ap.add_argument("--stand-hold-s", type=float, default=3.0)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dst / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    srcs = sorted(args.src.expanduser().glob("*.npz"))
    if not srcs and args.stand is None:
        raise SystemExit(f"no *.npz in {args.src}")
    for f in srcs:
        name = f"{f.stem}_hold.npz"
        manifest[name] = extend(f, args.dst / name, args.hold_s)
    if args.stand is not None:
        manifest["stand_hold.npz"] = extend(
            args.stand, args.dst / "stand_hold.npz", args.stand_hold_s
        )
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    print(f"manifest: {manifest_path} ({len(manifest)} clips)")


if __name__ == "__main__":
    main()
