"""Convert G1 29-DoF BONES-SEED CSV motions to H2 31-DoF tracking NPZ.

Pipeline (MuJoCo-only, mirrors the validated ``bones_seed_csv_to_npz.py``):

1. ``MotionLoader`` parses a G1 CSV clip (29 ``*_joint_dof`` cols + root), resamples
   120→50 fps, and computes root/joint velocities (slerp for root quat).
2. The 29 G1 joint angles are injected BY NAME into the H2 MuJoCo model. G1 and H2
   share identical joint names, so no remap is needed; H2's two extra joints
   (``head_pitch_joint``, ``head_yaw_joint``) are simply left at 0.
3. ``mj_forward`` runs H2 forward kinematics (joint ``range`` clamps applied), then
   we read back ALL 31 H2 hinge joints IN MODEL ORDER (= actuator/qpos order, the
   order the training env consumes) and every body's world pose/velocity via the
   injected ``track_*`` sensors (same as the G1 converter — no ``mj_objectVelocity``).

Output NPZ keys match the tracking loader exactly:
``fps, joint_pos(T,31), joint_vel(T,31), body_pos_w(T,nbody,3), body_quat_w(T,nbody,4),
body_lin_vel_w(T,nbody,3), body_ang_vel_w(T,nbody,3)``.

NAIVE RETARGET (first pass): G1 limb lengths ≠ H2, so injecting G1 joint angles does
not place H2 feet perfectly on the ground. ``--root_height_scale`` (default 1.31 =
H2 pelvis 1.04 / G1 pelvis 0.793) scales only root Z to reduce ground penetration.
Validate with ``scripts/motion/replay_npz.py`` before long training; a proper IK
retarget is a separate track.

Usage:
    uv run scripts/motion/g1_csv_to_h2_npz.py --input <csv_or_dir> --output <dir>
    uv run scripts/motion/g1_csv_to_h2_npz.py --input <dir> --output <dir> --limit 20
    uv run scripts/motion/g1_csv_to_h2_npz.py --input clip.csv --output out.npz --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.xml import inject_mujoco_tracking_sensors
from unilab.tools.bones_seed_csv import resolve_input_files

# Reuse the validated CSV parser / interpolator from the G1 converter (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bones_seed_csv_to_npz import MotionLoader  # noqa: E402

# H2 flat scene (built in src/unilab/assets/robots/h2/). Used for FK + tracking sensors.
DEFAULT_H2_MODEL = str(ASSETS_ROOT_PATH / "robots" / "h2" / "scene_flat.xml")
# H2 pelvis nominal 1.04 m vs G1 0.793 m -> naive root-Z scale to avoid sinking.
DEFAULT_ROOT_HEIGHT_SCALE = 1.04 / 0.793

_SENSOR_PREFIXES = ("track_pos_w_", "track_quat_w_", "track_linvel_w_", "track_angvel_w_")
_SENSOR_DIMS = (3, 4, 3, 3)


def _hinge_joint_readback(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """(qpos_adr, dof_adr) for every non-free joint, in model (tree/actuator) order."""
    out: list[tuple[int, int]] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        out.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))
    return out


def _map_csv_joints(model: mujoco.MjModel, joint_names: list[str]) -> list[tuple[int, int]]:
    """(qpos_adr, dof_adr) for each CSV joint, resolved by name in the H2 model."""
    inj: list[tuple[int, int]] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"CSV joint '{name}' not found in H2 model")
        inj.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))
    return inj


def _sensor_addr_table(model: mujoco.MjModel) -> np.ndarray:
    adrs = np.full((model.nbody, 4), -1, dtype=np.int32)
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        for k, prefix in enumerate(_SENSOR_PREFIXES):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}{body_name}")
            if sid >= 0:
                adrs[body_id, k] = model.sensor_adr[sid]
    return adrs


def run_h2_export(
    motion_loader: MotionLoader,
    h2_model_file: str,
    output_file: Path,
    root_height_scale: float,
) -> None:
    tmp_model_path, _, _ = inject_mujoco_tracking_sensors(h2_model_file)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_model_path)
    finally:
        Path(tmp_model_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)

    inj = _map_csv_joints(model, list(motion_loader.joint_names))  # 29 CSV joints
    readback = _hinge_joint_readback(model)                        # 31 H2 joints, model order
    num_frames = motion_loader.output_frames
    num_joints = len(readback)
    num_bodies = model.nbody

    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    sensor_adrs = _sensor_addr_table(model)

    for i in tqdm(range(num_frames), desc=output_file.stem, leave=False):
        data.qpos[0:3] = motion_loader.motion_base_poss[i]
        data.qpos[2] *= root_height_scale
        data.qpos[3:7] = motion_loader.motion_base_rots[i]
        data.qvel[0:3] = motion_loader.motion_base_lin_vels[i]
        data.qvel[3:6] = motion_loader.motion_base_ang_vels[i]

        for j, (qa, va) in enumerate(inj):
            data.qpos[qa] = motion_loader.motion_dof_poss[i, j]
            data.qvel[va] = motion_loader.motion_dof_vels[i, j]

        mujoco.mj_forward(model, data)

        for k, (qa, va) in enumerate(readback):
            joint_pos[i, k] = data.qpos[qa]
            joint_vel[i, k] = data.qvel[va]

        for body_id in range(num_bodies):
            pos_adr, quat_adr, lin_adr, ang_adr = sensor_adrs[body_id]
            if pos_adr >= 0:
                body_pos_w[i, body_id] = data.sensordata[pos_adr : pos_adr + _SENSOR_DIMS[0]]
            else:
                body_pos_w[i, body_id] = data.xpos[body_id]
            if quat_adr >= 0:
                body_quat_w[i, body_id] = data.sensordata[quat_adr : quat_adr + _SENSOR_DIMS[1]]
            else:
                body_quat_w[i, body_id] = data.xquat[body_id]
            if lin_adr >= 0:
                body_lin_vel_w[i, body_id] = data.sensordata[lin_adr : lin_adr + _SENSOR_DIMS[2]]
            if ang_adr >= 0:
                body_ang_vel_w[i, body_id] = data.sensordata[ang_adr : ang_adr + _SENSOR_DIMS[3]]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        fps=np.array([motion_loader.output_fps], dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert G1 CSV motions to H2 31-DoF NPZ")
    p.add_argument("--input", required=True, help="G1 CSV file or directory of CSV files")
    p.add_argument("--output", required=True, help="Output .npz (file input) or directory")
    p.add_argument("--h2_model_xml", default=DEFAULT_H2_MODEL, help="H2 scene_flat.xml")
    p.add_argument("--input_fps", type=float, default=120.0)
    p.add_argument("--output_fps", type=float, default=50.0)
    p.add_argument("--position_scale", type=float, default=0.01, help="cm->m on root translate")
    p.add_argument("--euler_order", type=str, default="xyz")
    p.add_argument(
        "--root_height_scale",
        type=float,
        default=DEFAULT_ROOT_HEIGHT_SCALE,
        help="naive root-Z scale G1->H2 (default 1.04/0.793≈1.31; 1.0 disables)",
    )
    p.add_argument("--limit", type=int, default=0, help="convert only the first N clips (0=all)")
    p.add_argument("--dry-run", action="store_true", help="validate/plan without writing NPZ")
    return p.parse_args()


def resolve_outputs(input_path: str, output: str, csv_files: list[Path]) -> list[Path]:
    in_root = Path(input_path).expanduser().resolve()
    if in_root.is_file():
        out = Path(output).expanduser().resolve()
        if out.suffix.lower() == ".npz":
            return [out]
        return [out / f"{in_root.stem}_h2.npz"]
    out_root = Path(output).expanduser().resolve()
    return [out_root / f"{f.stem}_h2.npz" for f in csv_files]


def main() -> None:
    args = parse_args()
    csv_files = resolve_input_files(args.input)
    if args.limit and args.limit > 0:
        csv_files = csv_files[: args.limit]
    output_files = resolve_outputs(args.input, args.output, csv_files)

    print(f"[g1_csv_to_h2_npz] {len(csv_files)} clip(s); H2 model: {args.h2_model_xml}")
    print(f"[g1_csv_to_h2_npz] input_fps={args.input_fps:g} output_fps={args.output_fps:g} "
          f"root_height_scale={args.root_height_scale:g}")
    if args.dry_run:
        print(f"[g1_csv_to_h2_npz] dry-run OK. Example output: {output_files[0]}")
        return

    failures: list[tuple[Path, str]] = []
    for csv_file, output_file in zip(csv_files, output_files, strict=True):
        try:
            loader = MotionLoader(
                motion_file=csv_file,
                input_fps=int(args.input_fps),
                output_fps=int(args.output_fps),
                position_scale=args.position_scale,
                euler_order=args.euler_order,
            )
            run_h2_export(loader, args.h2_model_xml, output_file, args.root_height_scale)
        except Exception as exc:  # keep going on a bad clip
            failures.append((csv_file, str(exc)))
            print(f"[g1_csv_to_h2_npz] FAILED {csv_file.name}: {exc}")

    ok = len(csv_files) - len(failures)
    print(f"[g1_csv_to_h2_npz] done: {ok}/{len(csv_files)} converted -> {output_files[0].parent}")
    if failures:
        print(f"[g1_csv_to_h2_npz] {len(failures)} failed")


if __name__ == "__main__":
    main()
