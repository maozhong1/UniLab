"""Render every NPZ motion in a directory into one MP4 (MuJoCo offscreen).

Clips are played back-to-back in sorted filename order. After each clip the
last pose is held for ``--gap`` seconds before the next clip starts.

Usage:
    uv run --no-sync python scripts/motion/replay_npz_dir.py \
        --npz_dir gr2/npz --output 12345.mp4 \
        --model_file src/unilab/assets/robots/gr2/scene_sonic_27dof.xml
"""

# pyright: reportAttributeAccessIssue=false

import argparse
import os
import sys
from pathlib import Path

# Offscreen rendering does not need a window; EGL works headless.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_npz import default_model_path, load_npz, resolve_motion_joint_names  # noqa: E402


def make_frame_setter(model: mujoco.MjModel, data: mujoco.MjData, motion: dict):
    """Return a callable that writes one motion frame into ``data``."""
    joint_pos = motion["joint_pos"]
    joint_vel = motion["joint_vel"]
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]
    joint_names = resolve_motion_joint_names(motion, model, joint_pos.shape[1])

    qpos_adr = []
    qvel_adr = []
    for name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise ValueError(f"Motion joint '{name}' is not found in model")
        qpos_adr.append(model.jnt_qposadr[jnt_id])
        qvel_adr.append(model.jnt_dofadr[jnt_id])
    qpos_adr = np.asarray(qpos_adr)
    qvel_adr = np.asarray(qvel_adr)

    root_body_id = 1  # floating base, same convention as replay_npz.py

    def set_frame(i: int) -> None:
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[0:3] = body_pos_w[i, root_body_id]
        data.qpos[3:7] = body_quat_w[i, root_body_id]
        data.qpos[qpos_adr] = joint_pos[i]
        data.qvel[qvel_adr] = joint_vel[i]
        mujoco.mj_forward(model, data)

    return set_frame


def main():
    parser = argparse.ArgumentParser(description="Render all NPZ motions in a directory to one MP4")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing *.npz motions")
    parser.add_argument("--output", type=str, required=True, help="Output MP4 path")
    parser.add_argument("--model_file", type=str, default=None, help="MuJoCo XML model file")
    parser.add_argument("--gap", type=float, default=5.0, help="Seconds to hold between clips")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--recursive", action="store_true", help="Search npz_dir recursively")
    parser.add_argument("--cam_distance", type=float, default=3.0)
    parser.add_argument("--cam_elevation", type=float, default=-15.0)
    parser.add_argument("--cam_azimuth", type=float, default=90.0)
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir).expanduser()
    pattern = "**/*.npz" if args.recursive else "*.npz"
    npz_files = sorted(npz_dir.glob(pattern))
    if not npz_files:
        print(f"Error: no NPZ files found in {npz_dir}")
        sys.exit(1)

    model_file = args.model_file or default_model_path()
    print(f"Model: {model_file}")
    print(f"Clips: {len(npz_files)} from {npz_dir}")
    model = mujoco.MjModel.from_xml_path(model_file)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1  # robot root body
    cam.distance = args.cam_distance
    cam.elevation = args.cam_elevation
    cam.azimuth = args.cam_azimuth

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output), fps=args.fps, codec="libx264", quality=8, macro_block_size=1
    )

    def render() -> np.ndarray:
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    gap_frames = int(round(args.gap * args.fps))
    total_frames = 0
    try:
        for clip_idx, npz_file in enumerate(npz_files):
            try:
                motion = load_npz(str(npz_file))
                set_frame = make_frame_setter(model, data, motion)
            except Exception as exc:  # skip malformed clips, keep the rest of the video
                print(f"[{clip_idx + 1}/{len(npz_files)}] skip {npz_file.name}: {exc}")
                continue

            num_frames = motion["joint_pos"].shape[0]
            duration = num_frames / motion["fps"] / args.speed
            out_frames = max(1, int(round(duration * args.fps)))
            print(f"[{clip_idx + 1}/{len(npz_files)}] {npz_file.name}: {num_frames} frames, {duration:.2f}s")

            # Resample motion time onto the output frame grid.
            for k in range(out_frames):
                src = min(num_frames - 1, int(k / args.fps * args.speed * motion["fps"]))
                set_frame(src)
                writer.append_data(render())
            total_frames += out_frames

            if clip_idx < len(npz_files) - 1 and gap_frames > 0:
                hold = render()  # last pose of this clip
                for _ in range(gap_frames):
                    writer.append_data(hold)
                total_frames += gap_frames
    finally:
        writer.close()
        renderer.close()

    print(f"Saved {output} ({total_frames} frames, {total_frames / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
