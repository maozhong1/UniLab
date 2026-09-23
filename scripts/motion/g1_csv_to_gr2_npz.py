"""Retarget G1 BONES-SEED CSV motions to a 27- or 29-DoF Fourier GR2.

G1 and GR2 joint columns are not interchangeable. G1 waist roll/pitch are dropped,
and task-space IK accounts for different link dimensions and body coordinate frames.

Usage:
    uv run --no-sync python scripts/motion/g1_csv_to_gr2_npz.py \
        --input clip.csv --output clip_gr2.npz --debug
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bones_seed_csv_to_npz import MotionLoader  # noqa: E402
from g1_csv_to_h2_npz import _SENSOR_DIMS, _sensor_addr_table  # noqa: E402

DEFAULT_G1_MODEL = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
DEFAULT_GR2_MODEL = str(ASSETS_ROOT_PATH / "robots" / "gr2" / "scene_flat.xml")

_BODY_MAP: tuple[tuple[str, str], ...] = (
    ("pelvis", "base"),
    ("left_hip_roll_link", "left_thigh_roll_link"),
    ("left_knee_link", "left_shank_pitch_link"),
    ("left_ankle_roll_link", "left_foot_roll_link"),
    ("right_hip_roll_link", "right_thigh_roll_link"),
    ("right_knee_link", "right_shank_pitch_link"),
    ("right_ankle_roll_link", "right_foot_roll_link"),
    ("torso_link", "waist_yaw_link"),
    ("left_shoulder_roll_link", "left_upper_arm_roll_link"),
    ("left_elbow_link", "left_lower_arm_pitch_link"),
    ("left_wrist_yaw_link", "left_hand_yaw_link"),
    ("right_shoulder_roll_link", "right_upper_arm_roll_link"),
    ("right_elbow_link", "right_lower_arm_pitch_link"),
    ("right_wrist_yaw_link", "right_hand_yaw_link"),
)
_ANCHOR_BODY = "base"
_FOOT_BODIES = ("left_foot_roll_link", "right_foot_roll_link")
_HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")

_POS_WEIGHTS = {
    "base": 8.0,
    "left_foot_roll_link": 12.0,
    "right_foot_roll_link": 12.0,
    "left_shank_pitch_link": 3.0,
    "right_shank_pitch_link": 3.0,
    "left_thigh_roll_link": 2.0,
    "right_thigh_roll_link": 2.0,
    "waist_yaw_link": 3.0,
    "left_hand_yaw_link": 2.0,
    "right_hand_yaw_link": 2.0,
    "left_lower_arm_pitch_link": 1.0,
    "right_lower_arm_pitch_link": 1.0,
    "left_upper_arm_roll_link": 1.5,
    "right_upper_arm_roll_link": 1.5,
}
_ORI_WEIGHTS = {
    "base": 8.0,
    "waist_yaw_link": 1.5,
    "left_foot_roll_link": 5.0,
    "right_foot_roll_link": 5.0,
}

# Target joint -> (G1 CSV joint, sign). GR2 elbow flexion uses the opposite sign.
_JOINT_MAP = {
    "waist_yaw_joint": ("waist_yaw_joint", 1.0),
    "left_shoulder_pitch_joint": ("left_shoulder_pitch_joint", 1.0),
    "left_shoulder_roll_joint": ("left_shoulder_roll_joint", 1.0),
    "left_shoulder_yaw_joint": ("left_shoulder_yaw_joint", 1.0),
    "left_elbow_pitch_joint": ("left_elbow_joint", -1.0),
    "left_wrist_yaw_joint": ("left_wrist_yaw_joint", 1.0),
    "left_wrist_pitch_joint": ("left_wrist_pitch_joint", 1.0),
    "left_wrist_roll_joint": ("left_wrist_roll_joint", 1.0),
    "right_shoulder_pitch_joint": ("right_shoulder_pitch_joint", 1.0),
    "right_shoulder_roll_joint": ("right_shoulder_roll_joint", 1.0),
    "right_shoulder_yaw_joint": ("right_shoulder_yaw_joint", 1.0),
    "right_elbow_pitch_joint": ("right_elbow_joint", -1.0),
    "right_wrist_yaw_joint": ("right_wrist_yaw_joint", 1.0),
    "right_wrist_pitch_joint": ("right_wrist_pitch_joint", 1.0),
    "right_wrist_roll_joint": ("right_wrist_roll_joint", 1.0),
    "left_hip_pitch_joint": ("left_hip_pitch_joint", 1.0),
    "left_hip_roll_joint": ("left_hip_roll_joint", 1.0),
    "left_hip_yaw_joint": ("left_hip_yaw_joint", 1.0),
    "left_knee_pitch_joint": ("left_knee_joint", 1.0),
    "left_ankle_pitch_joint": ("left_ankle_pitch_joint", 1.0),
    "left_ankle_roll_joint": ("left_ankle_roll_joint", 1.0),
    "right_hip_pitch_joint": ("right_hip_pitch_joint", 1.0),
    "right_hip_roll_joint": ("right_hip_roll_joint", 1.0),
    "right_hip_yaw_joint": ("right_hip_yaw_joint", 1.0),
    "right_knee_pitch_joint": ("right_knee_joint", 1.0),
    "right_ankle_pitch_joint": ("right_ankle_pitch_joint", 1.0),
    "right_ankle_roll_joint": ("right_ankle_roll_joint", 1.0),
}


def _body_ids(model: mujoco.MjModel, names) -> dict[str, int]:
    result = {}
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"body '{name}' not found in model")
        result[name] = body_id
    return result


def _scalar_joints(model: mujoco.MjModel) -> list[tuple[str, int, int, int]]:
    joints = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise ValueError(f"joint id {joint_id} has no name")
        joints.append(
            (name, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]), joint_id)
        )
    return sorted(joints, key=lambda item: item[1])


def _validate_target(model: mujoco.MjModel) -> list[tuple[str, int, int, int]]:
    joints = _scalar_joints(model)
    names = [joint[0] for joint in joints]
    actuator_names = [
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[index, 0])
        )
        for index in range(model.nu)
    ]
    if len(joints) != model.nu or len(joints) not in (27, 29):
        raise ValueError(f"GR2 must have 27 or 29 joints/actuators, got {len(joints)}/{model.nu}")
    if actuator_names != names:
        raise ValueError("GR2 actuator order must match scalar qpos order")
    has_head = set(_HEAD_JOINTS).issubset(names)
    if has_head != (len(joints) == 29):
        raise ValueError("GR2 head joint presence does not match the 27/29-DoF profile")
    if {"waist_roll_joint", "waist_pitch_joint"}.intersection(names):
        raise ValueError("GR2 unexpectedly contains waist roll/pitch joints")
    return joints


def _quat_inverse(quaternion: np.ndarray) -> np.ndarray:
    result = np.empty(4)
    mujoco.mju_negQuat(result, quaternion)
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(4)
    mujoco.mju_mulQuat(result, left, right)
    return result


def _quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = _quat_multiply(target, _quat_inverse(current))
    if delta[0] < 0.0:
        delta = -delta
    sin_half = np.linalg.norm(delta[1:])
    if sin_half < 1e-9:
        return np.zeros(3)
    return delta[1:] * (2.0 * np.arctan2(sin_half, delta[0]) / sin_half)


class SourceFK:
    def __init__(self, model_file: str, joint_names: list[str]):
        self.model = mujoco.MjModel.from_xml_path(model_file)
        self.data = mujoco.MjData(self.model)
        self.joint_addresses = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"CSV joint '{name}' not found in G1 model")
            self.joint_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        self.body_ids = _body_ids(self.model, [source for source, _target in _BODY_MAP])
        self.data.qpos[:] = self.model.qpos0
        mujoco.mj_forward(self.model, self.data)
        self.rest_quat = {
            target: self.data.xquat[self.body_ids[source]].copy()
            for source, target in _BODY_MAP
        }

    def poses(self, root_pos, root_quat, joint_pos):
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_quat
        for value, address in zip(joint_pos, self.joint_addresses, strict=True):
            self.data.qpos[address] = value
        mujoco.mj_forward(self.model, self.data)
        positions = {
            target: self.data.xpos[self.body_ids[source]].copy()
            for source, target in _BODY_MAP
        }
        quaternions = {
            target: self.data.xquat[self.body_ids[source]].copy()
            for source, target in _BODY_MAP
        }
        return positions, quaternions


def _standing_geometry(model_file: str, anchor: str, hip: str, foot: str):
    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ids = _body_ids(model, (anchor, hip, foot))
    anchor_z = float(data.xpos[ids[anchor], 2])
    foot_z = float(data.xpos[ids[foot], 2])
    return anchor_z, float(data.xpos[ids[hip], 2] - foot_z), foot_z


def _build_targets(
    source_pos,
    source_quat,
    source_rest_quat,
    target_rest_quat,
    horizontal_scale,
    vertical_scale,
    foot_offset,
):
    support_z = min(source_pos[name][2] for name in _FOOT_BODIES)
    anchor_xy = source_pos[_ANCHOR_BODY][:2]
    target_pos = {}
    for name, position in source_pos.items():
        xy = anchor_xy + (position[:2] - anchor_xy) * horizontal_scale
        z = (position[2] - support_z) * vertical_scale + foot_offset
        target_pos[name] = np.array((xy[0], xy[1], z))
    target_quat = {}
    for name in _ORI_WEIGHTS:
        source_delta = _quat_multiply(source_quat[name], _quat_inverse(source_rest_quat[name]))
        target_quat[name] = _quat_multiply(source_delta, target_rest_quat[name])
    return target_pos, target_quat


class GR2IK:
    def __init__(self, model, data, damping: float, posture_weight: float):
        self.model = model
        self.data = data
        self.damping = damping
        self.posture_weight = posture_weight
        self.joints = _validate_target(model)
        self.body_ids = _body_ids(model, _POS_WEIGHTS)
        joint_by_name = {joint[0]: joint for joint in self.joints}
        self.head_addresses = [
            joint_by_name[name][1] for name in _HEAD_JOINTS if name in joint_by_name
        ]
        self.ranges = np.asarray(
            [model.jnt_range[joint_id] for _name, _qa, _va, joint_id in self.joints]
        )
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros((3, model.nv))
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        self.rest_quat = {
            name: data.xquat[self.body_ids[name]].copy() for name in _ORI_WEIGHTS
        }

    def _clamp(self):
        for index, (_name, qpos_address, _dof_address, _joint_id) in enumerate(self.joints):
            low, high = self.ranges[index]
            if low < high:
                self.data.qpos[qpos_address] = np.clip(self.data.qpos[qpos_address], low, high)
        for address in self.head_addresses:
            self.data.qpos[address] = 0.0

    def solve(self, target_pos, target_quat, posture_bias, iterations: int):
        rows = 3 * len(_POS_WEIGHTS) + 3 * len(_ORI_WEIGHTS) + len(self.joints)
        jacobian = np.zeros((rows, self.model.nv))
        error = np.zeros(rows)
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            row = 0
            for name, weight in _POS_WEIGHTS.items():
                body_id = self.body_ids[name]
                mujoco.mj_jacBody(
                    self.model, self.data, self.jac_pos, self.jac_rot, body_id
                )
                jacobian[row : row + 3] = weight * self.jac_pos
                error[row : row + 3] = weight * (
                    target_pos[name] - self.data.xpos[body_id]
                )
                row += 3
            for name, weight in _ORI_WEIGHTS.items():
                body_id = self.body_ids[name]
                mujoco.mj_jacBody(
                    self.model, self.data, self.jac_pos, self.jac_rot, body_id
                )
                jacobian[row : row + 3] = weight * self.jac_rot
                error[row : row + 3] = weight * _quat_error(
                    target_quat[name], self.data.xquat[body_id]
                )
                row += 3
            for index, (_name, qpos_address, dof_address, _joint_id) in enumerate(self.joints):
                jacobian[row, dof_address] = self.posture_weight
                error[row] = self.posture_weight * (
                    posture_bias[index] - self.data.qpos[qpos_address]
                )
                row += 1
            normal = jacobian.T @ jacobian
            normal[np.diag_indices_from(normal)] += self.damping**2
            delta_q = np.linalg.solve(normal, jacobian.T @ error)
            mujoco.mj_integratePos(self.model, self.data.qpos, delta_q, 1.0)
            self._clamp()


def _posture_bias(target_joints, source_names, source_positions):
    source_index = {name: index for index, name in enumerate(source_names)}
    bias = np.zeros(len(target_joints))
    for target_index, (target_name, _qa, _va, _joint_id) in enumerate(target_joints):
        mapping = _JOINT_MAP.get(target_name)
        if mapping is None:
            continue
        source_name, sign = mapping
        if source_name not in source_index:
            raise ValueError(f"required G1 joint '{source_name}' missing from CSV")
        bias[target_index] = sign * source_positions[source_index[source_name]]
    return bias


def _resolve_outputs(input_path: str, output: str, csv_files: list[Path]) -> list[Path]:
    source = Path(input_path).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    if source.is_file():
        return [target] if target.suffix == ".npz" else [target / f"{source.stem}_gr2.npz"]
    return [target / f"{csv_file.stem}_gr2.npz" for csv_file in csv_files]


def run_export(
    loader,
    source,
    model_file,
    output_file,
    horizontal_scale,
    vertical_scale,
    foot_offset,
    damping,
    posture_weight,
    iterations,
    warm_iterations,
    debug,
    ground,
    ground_tolerance,
):
    temporary_model, _, _ = inject_mujoco_tracking_sensors(model_file)
    try:
        model = mujoco.MjModel.from_xml_path(temporary_model)
    finally:
        Path(temporary_model).unlink(missing_ok=True)
    data = mujoco.MjData(model)
    ik = GR2IK(model, data, damping, posture_weight)
    sensors = _sensor_addr_table(model)
    joint_names = [joint[0] for joint in ik.joints]
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
        for body_id in range(model.nbody)
    ]
    foot_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id]
        and (
            body_name := mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
            )
        )
        and "foot" in body_name
    }

    frames = loader.output_frames
    qpos = np.zeros((frames, model.nq))
    data.qpos[:] = model.qpos0
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    foot_error = 0.0
    for frame in tqdm(range(frames), desc=f"{output_file.stem} ik", leave=False):
        source_pos, source_quat = source.poses(
            loader.motion_base_poss[frame],
            loader.motion_base_rots[frame],
            loader.motion_dof_poss[frame],
        )
        target_pos, target_quat = _build_targets(
            source_pos,
            source_quat,
            source.rest_quat,
            ik.rest_quat,
            horizontal_scale,
            vertical_scale,
            foot_offset,
        )
        bias = _posture_bias(ik.joints, list(loader.joint_names), loader.motion_dof_poss[frame])
        if frame == 0:
            data.qpos[:3] = target_pos[_ANCHOR_BODY]
            data.qpos[3:7] = target_quat[_ANCHOR_BODY]
        ik.solve(target_pos, target_quat, bias, warm_iterations if frame == 0 else iterations)
        mujoco.mj_forward(model, data)
        qpos[frame] = data.qpos
        if debug:
            foot_error += sum(
                np.linalg.norm(target_pos[name] - data.xpos[ik.body_ids[name]])
                for name in _FOOT_BODIES
            )

    lifted = 0
    maximum_penetration = 0.0
    if ground:
        for frame in range(frames):
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            penetration = min(
                [
                    float(data.contact[index].dist)
                    for index in range(data.ncon)
                    if data.contact[index].geom1 in foot_geoms
                    or data.contact[index].geom2 in foot_geoms
                ]
                or [0.0]
            )
            if penetration < -ground_tolerance:
                qpos[frame, 2] -= penetration
                lifted += 1
                maximum_penetration = min(maximum_penetration, penetration)

    dt = 1.0 / loader.output_fps
    qvel = np.zeros((frames, model.nv))
    for frame in range(frames):
        previous, following = max(frame - 1, 0), min(frame + 1, frames - 1)
        span = (following - previous) * dt
        if span:
            mujoco.mj_differentiatePos(model, qvel[frame], span, qpos[previous], qpos[following])

    joint_pos = np.zeros((frames, len(ik.joints)), dtype=np.float32)
    joint_vel = np.zeros_like(joint_pos)
    body_pos = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    body_quat = np.zeros((frames, model.nbody, 4), dtype=np.float32)
    body_linvel = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    body_angvel = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    for frame in range(frames):
        data.qpos[:], data.qvel[:] = qpos[frame], qvel[frame]
        mujoco.mj_forward(model, data)
        for index, (_name, qpos_address, dof_address, _joint_id) in enumerate(ik.joints):
            joint_pos[frame, index] = data.qpos[qpos_address]
            joint_vel[frame, index] = data.qvel[dof_address]
        for body_id in range(model.nbody):
            pos_address, quat_address, lin_address, ang_address = sensors[body_id]
            body_pos[frame, body_id] = (
                data.sensordata[pos_address : pos_address + _SENSOR_DIMS[0]]
                if pos_address >= 0
                else data.xpos[body_id]
            )
            body_quat[frame, body_id] = (
                data.sensordata[quat_address : quat_address + _SENSOR_DIMS[1]]
                if quat_address >= 0
                else data.xquat[body_id]
            )
            if lin_address >= 0:
                body_linvel[frame, body_id] = data.sensordata[lin_address : lin_address + 3]
            if ang_address >= 0:
                body_angvel[frame, body_id] = data.sensordata[ang_address : ang_address + 3]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        fps=np.asarray([loader.output_fps], dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_linvel,
        body_ang_vel_w=body_angvel,
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
    )
    return foot_error / max(2 * frames, 1), lifted, -maximum_penetration


def parse_args():
    parser = argparse.ArgumentParser(description="IK-retarget G1 CSV motions to GR2")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--g1_model_xml", default=DEFAULT_G1_MODEL)
    parser.add_argument("--gr2_model_xml", default=DEFAULT_GR2_MODEL)
    parser.add_argument("--input_fps", type=float, default=120.0)
    parser.add_argument("--output_fps", type=float, default=50.0)
    parser.add_argument("--position_scale", type=float, default=0.01)
    parser.add_argument("--euler_order", default="xyz")
    parser.add_argument("--h_scale", type=float, default=0.0)
    parser.add_argument("--v_scale", type=float, default=0.0)
    parser.add_argument("--ik_iters", type=int, default=30)
    parser.add_argument("--warm_iters", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--posture_w", type=float, default=0.1)
    parser.add_argument("--no_ground", action="store_true")
    parser.add_argument("--ground_tol", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_files = resolve_input_files(args.input)
    if args.limit > 0:
        csv_files = csv_files[: args.limit]
    output_files = _resolve_outputs(args.input, args.output, csv_files)
    g1_anchor, g1_leg, _ = _standing_geometry(
        args.g1_model_xml, "pelvis", "left_hip_roll_link", "left_ankle_roll_link"
    )
    gr2_anchor, gr2_leg, gr2_foot = _standing_geometry(
        args.gr2_model_xml, "base", "left_thigh_roll_link", "left_foot_roll_link"
    )
    horizontal_scale = args.h_scale or gr2_leg / g1_leg
    vertical_scale = args.v_scale or gr2_anchor / g1_anchor
    print(f"[g1_csv_to_gr2_npz] {len(csv_files)} clip(s)")
    print(f"[g1_csv_to_gr2_npz] G1={args.g1_model_xml}")
    print(f"[g1_csv_to_gr2_npz] GR2={args.gr2_model_xml}")
    print(
        f"[g1_csv_to_gr2_npz] scales: h={horizontal_scale:.4f} "
        f"v={vertical_scale:.4f} foot={gr2_foot:.4f}"
    )
    print(
        f"[g1_csv_to_gr2_npz] IK: iterations={args.ik_iters}, "
        f"warm={args.warm_iters}, damping={args.damping}"
    )
    if args.dry_run:
        _validate_target(mujoco.MjModel.from_xml_path(args.gr2_model_xml))
        print(f"[g1_csv_to_gr2_npz] dry-run OK: {output_files[0]}")
        return

    failures = []
    skipped = 0
    for csv_file, output_file in zip(csv_files, output_files, strict=True):
        if args.skip_existing and output_file.exists():
            skipped += 1
            continue
        try:
            loader = MotionLoader(
                motion_file=csv_file,
                input_fps=int(args.input_fps),
                output_fps=int(args.output_fps),
                position_scale=args.position_scale,
                euler_order=args.euler_order,
            )
            stats = run_export(
                loader,
                SourceFK(args.g1_model_xml, list(loader.joint_names)),
                args.gr2_model_xml,
                output_file,
                horizontal_scale,
                vertical_scale,
                gr2_foot,
                args.damping,
                args.posture_w,
                args.ik_iters,
                args.warm_iters,
                args.debug,
                not args.no_ground,
                args.ground_tol,
            )
            if args.debug:
                print(
                    f"[g1_csv_to_gr2_npz] {output_file.name}: "
                    f"mean_foot_error={stats[0]:.4f} m, grounded={stats[1]}/"
                    f"{loader.output_frames}, max_penetration={stats[2]:.4f} m"
                )
        except Exception as error:
            failures.append((csv_file, str(error)))
            print(f"[g1_csv_to_gr2_npz] FAILED {csv_file.name}: {error}")

    converted = len(csv_files) - skipped - len(failures)
    print(f"[g1_csv_to_gr2_npz] done: {converted}/{len(csv_files)} converted")
    if failures:
        for csv_file, error in failures:
            print(f"  {csv_file}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
