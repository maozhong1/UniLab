#!/usr/bin/env python3
"""Filter H2 motion npz clips by angular-velocity "violence" for curriculum training.

Why: the H2 SONIC clips are dance motions with fast spins (chaines / 180-turns) whose
reference body angular velocity peaks at 8-19 rad/s. A torque/bandwidth-limited humanoid
cannot reproduce those spins while staying upright, so motion_body_ang_vel saturates at
its "robot barely rotates" floor (~0.16) regardless of reward/contact tuning. Training a
curriculum stage on the *calm* subset lifts ang_vel tracking and stabilizes learning far
more than any reward knob.

This tool scans a directory of *.npz clips, computes a per-clip angular-velocity metric
over ``body_ang_vel_w`` (all bodies, all frames), prints the distribution so you can pick
a threshold, and materializes a "calm" subset as symlinks (default) or copies. The
training launcher just points NPZ_DIR at the output dir.

Examples:
  # 1) just report the distribution, pick a threshold:
  uv run --no-sync python scripts/motion/filter_calm_h2.py --src /path/npz_h2_new --dry-run
  # 2) materialize clips with per-clip p99 |w| <= 8 rad/s as symlinks:
  uv run --no-sync python scripts/motion/filter_calm_h2.py \
      --src /path/npz_h2_new --dst /path/npz_h2_calm --metric p99 --thresh 8.0
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

import numpy as np

_METRICS = ("p99", "p95", "max", "mean")


def clip_metric(ang_vel_w: np.ndarray, metric: str) -> float:
    """Reduce (T, B, 3) angular velocity to one scalar |w| statistic (rad/s)."""
    mag = np.linalg.norm(ang_vel_w, axis=-1)  # (T, B) per-body-per-frame speed
    if metric == "max":
        return float(mag.max())
    if metric == "mean":
        return float(mag.mean())
    pct = 99.0 if metric == "p99" else 95.0
    return float(np.percentile(mag, pct))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir of source *.npz clips")
    ap.add_argument("--dst", default=None, help="output dir for the calm subset (default: <src>_calm)")
    ap.add_argument("--metric", default="p99", choices=_METRICS, help="per-clip |w| statistic (default p99)")
    ap.add_argument("--thresh", type=float, default=8.0, help="keep clips with metric <= thresh rad/s (default 8.0)")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    ap.add_argument("--dry-run", action="store_true", help="report distribution only; do not write anything")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    files = sorted(glob.glob(str(src / "*.npz")))
    if not files:
        raise SystemExit(f"no *.npz under {src}")

    # ── scan ──────────────────────────────────────────────────────────
    vals: list[float] = []
    names: list[str] = []
    skipped: list[str] = []
    for f in files:
        try:
            z = np.load(f)
            if "body_ang_vel_w" not in z:
                skipped.append(f)
                continue
            vals.append(clip_metric(np.asarray(z["body_ang_vel_w"], dtype=np.float32), args.metric))
            names.append(f)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  ! skip {os.path.basename(f)}: {e}")
            skipped.append(f)

    v = np.asarray(vals)
    n = v.size
    print(f"\nscanned {n} clips from {src}  (skipped {len(skipped)})")
    print(f"per-clip metric = {args.metric} of body |w| (rad/s)")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("  distribution:", {f"p{q}": round(float(np.percentile(v, q)), 2) for q in qs})
    print(f"  min={v.min():.2f}  max={v.max():.2f}  mean={v.mean():.2f}")

    keep_mask = v <= args.thresh
    n_keep = int(keep_mask.sum())
    print(f"\nthreshold {args.metric} <= {args.thresh} rad/s  ->  keep {n_keep}/{n} "
          f"({100.0 * n_keep / n:.1f}%), drop {n - n_keep}")
    # show a few of the most-violent dropped clips (the ceiling-killers)
    drop_order = np.argsort(-v)
    print("  most-violent dropped (top 5):")
    shown = 0
    for i in drop_order:
        if not keep_mask[i]:
            print(f"    {v[i]:6.2f}  {os.path.basename(names[i])}")
            shown += 1
            if shown >= 5:
                break

    if args.dry_run:
        print("\n[dry-run] nothing written. Re-run without --dry-run to materialize the subset.")
        return
    if n_keep == 0:
        raise SystemExit("threshold keeps 0 clips; raise --thresh")

    dst = Path(args.dst) if args.dst else src.parent / f"{src.name}_calm"
    dst = dst.expanduser().resolve()
    if dst.exists():
        # only clear symlinks/files we would have made; refuse to clobber a non-empty real dir
        existing = list(dst.glob("*.npz"))
        for p in existing:
            if p.is_symlink() or args.copy:
                p.unlink()
    dst.mkdir(parents=True, exist_ok=True)

    for i in np.nonzero(keep_mask)[0]:
        srcf = Path(names[i])
        dstf = dst / srcf.name
        if dstf.exists() or dstf.is_symlink():
            dstf.unlink()
        if args.copy:
            shutil.copy2(srcf, dstf)
        else:
            dstf.symlink_to(srcf)

    mode = "copied" if args.copy else "symlinked"
    print(f"\n{mode} {n_keep} clips -> {dst}")
    print("point the training launcher at it:")
    print(f"  NPZ_DIR={dst}")


if __name__ == "__main__":
    main()
