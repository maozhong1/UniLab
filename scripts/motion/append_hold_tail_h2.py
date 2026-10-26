#!/usr/bin/env python3
"""Append a static hold of the last frame to H2 motion clips that end in double support.

Deploy holds a clip's last frame (zero velocity) after it ends, but training truncates the
episode at the clip end, so "stand still on the final pose" is never trained and the H2
policy drifts and steps during the hold.
This writes, for every clip whose last frame has both feet planted, a copy with
``--hold-s`` seconds of that frame appended (joint/body velocities zero). Other clips are
symlinked unchanged (a single-support final pose cannot be held).

Inputs are H2 NPZ files from ``g1_csv_to_h2_npz2.py`` (body arrays indexed by MuJoCo body
id, resolved once from the H2 model on this cold path).

Example:
  uv run --no-sync python scripts/motion/append_hold_tail_h2.py \
      --src /tmp/sonic_train/h2_stage2_95 --dst /tmp/sonic_train/h2_stage2_95_hold --hold-s 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH

DEFAULT_H2_MODEL = str(ASSETS_ROOT_PATH / "robots" / "h2" / "scene_flat.xml")
_FOOT_BODIES = ("left_ankle_pitch_link", "right_ankle_pitch_link")
_VEL_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")


def _foot_ids(model_path: str) -> list[int]:
    model = mujoco.MjModel.from_xml_path(model_path)
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in _FOOT_BODIES]


def ends_in_double_support(clip, foot_ids, height_tol: float, max_speed: float) -> bool:
    """Both feet within ``height_tol`` of their clip-lowest height and slower than ``max_speed``."""
    pos = clip["body_pos_w"][:, foot_ids]          # (T, 2, 3)
    vel = clip["body_lin_vel_w"][-5:, foot_ids]    # last 0.1 s at 50 fps
    lift = pos[-1, :, 2] - pos[:, :, 2].min(axis=0)
    speed = np.linalg.norm(vel[..., :2], axis=-1).max(axis=0)
    return bool(np.all(lift < height_tol) and np.all(speed < max_speed))


def with_hold(clip, hold_s: float) -> dict[str, np.ndarray]:
    n = int(round(hold_s * int(clip["fps"][0])))
    out = {}
    for key in clip.files:
        arr = clip[key]
        if key == "fps":
            out[key] = arr
            continue
        tail = np.repeat(arr[-1:], n, axis=0)
        if key in _VEL_KEYS:
            tail = np.zeros_like(tail)
        out[key] = np.concatenate([arr, tail], axis=0).astype(arr.dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--height-tol", type=float, default=0.03, help="m above the clip-lowest foot height")
    ap.add_argument("--max-speed", type=float, default=0.15, help="m/s horizontal foot speed")
    ap.add_argument("--model", default=DEFAULT_H2_MODEL)
    args = ap.parse_args()

    foot_ids = _foot_ids(args.model)
    args.dst.mkdir(parents=True, exist_ok=True)
    for existing in args.dst.glob("*.npz"):
        existing.unlink()
    held = 0
    sources = sorted(args.src.glob("*.npz"))
    for src in sources:
        target = args.dst / src.name
        with np.load(src) as clip:
            if ends_in_double_support(clip, foot_ids, args.height_tol, args.max_speed):
                np.savez(target, **with_hold(clip, args.hold_s))
                held += 1
                continue
        target.symlink_to(src.resolve())
    print(f"appended a {args.hold_s:g} s hold to {held}/{len(sources)} clips -> {args.dst}")


if __name__ == "__main__":
    main()
