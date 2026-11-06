"""Retarget G1 BONES-SEED CSV motions to a 27- or 29-DoF Fourier GR2.

G1 and GR2 joint columns are not interchangeable. G1 waist roll/pitch are dropped,
and task-space IK accounts for different link dimensions and body coordinate frames.

Arm self-collision handling (on by default):
  * arm width shift: G1 shoulders are narrower than GR2's, so arm targets are shifted
    outward by the shoulder-width difference, faded out as the hand nears the midline
    (keeps clapping / two-hand holds intact while moving swinging hands off the hip motors).
  * collision avoidance: after the regular IK solve, any arm-vs-trunk/leg/other-arm pair
    closer than the safe distance triggers extra IK iterations that only move the shoulder,
    elbow and wrist joints, so root, legs and feet stay as solved without the constraint.

Usage:
    uv run --no-sync python scripts/motion/g1_csv_to_gr2_npz.py \
        --input clip.csv --output clip_gr2.npz --debug
    uv run --no-sync python scripts/motion/g1_csv_to_gr2_npz.py \
        --input <csv_dir> --output <npz_dir> \
        --gr2_model_xml src/unilab/assets/robots/gr2/scene_sonic_27dof.xml --workers 16
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One BLAS thread per process: the IK solves are tiny and --workers parallelizes across clips.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from unilab.assets import ASSETS_ROOT_PATH  # noqa: E402
from unilab.base.backend.mujoco.xml import inject_mujoco_tracking_sensors  # noqa: E402
from unilab.tools.bones_seed_csv import resolve_input_files  # noqa: E402

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

# Arm chain bodies shifted outward by the shoulder-width difference, with their side sign.
_ARM_SHIFT_BODIES = {
    "left_upper_arm_roll_link": 1.0,
    "left_lower_arm_pitch_link": 1.0,
    "left_hand_yaw_link": 1.0,
    "right_upper_arm_roll_link": -1.0,
    "right_lower_arm_pitch_link": -1.0,
    "right_hand_yaw_link": -1.0,
}
_TORSO_BODY = "waist_yaw_link"
_SHOULDER_BODIES = ("left_shoulder_roll_link", "left_upper_arm_roll_link")  # (G1, GR2)

# Self-collision groups. Shoulder links are left out: they always touch the torso.
_COLLISION_GROUPS = {
    "L_arm": (
        "left_upper_arm_yaw_link", "left_lower_arm_pitch_link", "left_hand_yaw_link",
        "left_hand_pitch_link", "left_hand_roll_link",
    ),
    "R_arm": (
        "right_upper_arm_yaw_link", "right_lower_arm_pitch_link", "right_hand_yaw_link",
        "right_hand_pitch_link", "right_hand_roll_link",
    ),
    "trunk": ("base", "waist_yaw_link", "head_yaw_link", "head_pitch_link"),
    "L_leg": (
        "left_thigh_pitch_link", "left_thigh_roll_link", "left_thigh_yaw_link",
        "left_shank_pitch_link",
    ),
    "R_leg": (
        "right_thigh_pitch_link", "right_thigh_roll_link", "right_thigh_yaw_link",
        "right_shank_pitch_link",
    ),
}
_GROUP_BIT = {name: 1 << index for index, name in enumerate(_COLLISION_GROUPS)}
_ARM_GROUPS = ("L_arm", "R_arm")
_ARM_JOINT_KEYS = ("shoulder", "elbow", "wrist")
_HAND_BODY = "hand_roll_link"
_PENETRATION_REPORT = 0.01
_REPAIR_HOLD_WEIGHT = 3.0


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


class SelfCollision:
    """Arm proximity queries on a collision-only copy of the GR2 model (built once per clip).

    Only arm-vs-trunk, arm-vs-leg and arm-vs-arm pairs are enabled. Link meshes collide as
    convex hulls, matching the self-penetration acceptance scan.
    """

    def __init__(self, model_file, body_ids, body_dist, arm_dist, ramp, hand_inflate):
        spec = mujoco.MjSpec.from_file(model_file)
        group_of = {body: group for group, bodies in _COLLISION_GROUPS.items() for body in bodies}
        self.body_dist = body_dist
        self.arm_dist = arm_dist
        self.ramp = max(ramp, 1e-6)
        for geom in spec.geoms:
            group = group_of.get(geom.parent.name) if geom.parent else None
            enabled = group is not None and geom.type in (
                mujoco.mjtGeom.mjGEOM_MESH,
                mujoco.mjtGeom.mjGEOM_BOX,
            )
            bit = _GROUP_BIT[group] if enabled else 0
            if group in _ARM_GROUPS:
                affinity = sum(_GROUP_BIT.values()) & ~bit
            else:
                affinity = sum(_GROUP_BIT[arm] for arm in _ARM_GROUPS) if enabled else 0
            geom.contype = bit
            geom.conaffinity = affinity
            geom.margin = max(body_dist, arm_dist) + self.ramp if enabled else 0.0
            geom.gap = 0.0
            if enabled and hand_inflate and geom.type == mujoco.mjtGeom.mjGEOM_BOX:
                if geom.parent.name.endswith(_HAND_BODY):
                    geom.size = np.asarray(geom.size) + hand_inflate
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            for body in range(self.model.nbody)
        ]
        self.geom_group = [group_of.get(names[self.model.geom_bodyid[geom]]) for geom in range(self.model.ngeom)]
        # Contacts are reported on this model; Jacobians use the IK model's body ids.
        self.geom_body = [body_ids.get(names[self.model.geom_bodyid[geom]], -1) for geom in range(self.model.ngeom)]

    def query(self, qpos, include_ramp=True):
        """Closest point per geom pair: (body1, body2, pos, normal 1->2, dist, safe)."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        closest = {}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = (contact.geom1, contact.geom2)
            if pair not in closest or contact.dist < closest[pair].dist:
                closest[pair] = contact
        result = []
        for (geom1, geom2), contact in closest.items():
            both_arms = self.geom_group[geom1] in _ARM_GROUPS and self.geom_group[geom2] in _ARM_GROUPS
            safe = self.arm_dist if both_arms else self.body_dist
            if contact.dist >= safe + (self.ramp if include_ramp else 0.0):
                continue
            result.append(
                (
                    self.geom_body[geom1],
                    self.geom_body[geom2],
                    contact.pos.copy(),
                    contact.frame[:3].copy(),
                    float(contact.dist),
                    safe,
                )
            )
        return result

    def max_penetration(self, qpos):
        """Deepest arm penetration (m, >= 0) for one configuration."""
        return max([-entry[4] for entry in self.query(qpos, include_ramp=False)] + [0.0])


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
    arm_shift=0.0,
    shoulder_half_width=1.0,
):
    support_z = min(source_pos[name][2] for name in _FOOT_BODIES)
    anchor_xy = source_pos[_ANCHOR_BODY][:2]
    target_pos = {}
    for name, position in source_pos.items():
        xy = anchor_xy + (position[:2] - anchor_xy) * horizontal_scale
        z = (position[2] - support_z) * vertical_scale + foot_offset
        target_pos[name] = np.array((xy[0], xy[1], z))
    if arm_shift:
        torso_rotation = np.empty(9)
        mujoco.mju_quat2Mat(torso_rotation, source_quat[_TORSO_BODY])
        lateral_axis = torso_rotation.reshape(3, 3)[:, 1]
        for name, side in _ARM_SHIFT_BODIES.items():
            lateral = side * np.dot(source_pos[name] - source_pos[_TORSO_BODY], lateral_axis)
            weight = np.clip(lateral / shoulder_half_width, 0.0, 1.0)
            target_pos[name] = target_pos[name] + side * arm_shift * weight * lateral_axis
    target_quat = {}
    for name in _ORI_WEIGHTS:
        source_delta = _quat_multiply(source_quat[name], _quat_inverse(source_rest_quat[name]))
        target_quat[name] = _quat_multiply(source_delta, target_rest_quat[name])
    return target_pos, target_quat


def _gaussian_filter(values, sigma):
    """Gaussian low-pass along axis 0 with edge padding."""
    radius = max(int(3.0 * sigma + 0.5), 1)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])],
        axis=1,
    )


def _arm_width_shift(g1_model_file, gr2_model_file, horizontal_scale):
    """(outward shift, G1 shoulder half width) from the two rest poses (cold path)."""
    widths = []
    for model_file, shoulder in zip((g1_model_file, gr2_model_file), _SHOULDER_BODIES, strict=True):
        model = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        torso = "torso_link" if shoulder == _SHOULDER_BODIES[0] else _TORSO_BODY
        ids = _body_ids(model, (torso, shoulder))
        widths.append(float(data.xpos[ids[shoulder], 1] - data.xpos[ids[torso], 1]))
    g1_width, gr2_width = widths
    return gr2_width - horizontal_scale * g1_width, g1_width


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
        self.arm_dofs = np.asarray(
            [
                dof_address
                for name, _qa, dof_address, _joint_id in self.joints
                if any(key in name for key in _ARM_JOINT_KEYS)
            ]
        )
        self.arm_qpos = np.asarray(
            [
                qpos_address
                for name, qpos_address, _va, _joint_id in self.joints
                if any(key in name for key in _ARM_JOINT_KEYS)
            ]
        )
        self.jac_other = np.zeros((3, model.nv))

    def _clamp(self):
        for index, (_name, qpos_address, _dof_address, _joint_id) in enumerate(self.joints):
            low, high = self.ranges[index]
            if low < high:
                self.data.qpos[qpos_address] = np.clip(self.data.qpos[qpos_address], low, high)
        for address in self.head_addresses:
            self.data.qpos[address] = 0.0

    def _system(self, target_pos, target_quat, posture_bias):
        rows = 3 * len(_POS_WEIGHTS) + 3 * len(_ORI_WEIGHTS) + len(self.joints)
        jacobian = np.zeros((rows, self.model.nv))
        error = np.zeros(rows)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        row = 0
        for name, weight in _POS_WEIGHTS.items():
            body_id = self.body_ids[name]
            mujoco.mj_jacBody(self.model, self.data, self.jac_pos, self.jac_rot, body_id)
            jacobian[row : row + 3] = weight * self.jac_pos
            error[row : row + 3] = weight * (target_pos[name] - self.data.xpos[body_id])
            row += 3
        for name, weight in _ORI_WEIGHTS.items():
            body_id = self.body_ids[name]
            mujoco.mj_jacBody(self.model, self.data, self.jac_pos, self.jac_rot, body_id)
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
        return jacobian, error

    def _step(self, jacobian, error, damping, dofs=None):
        if dofs is not None:
            jacobian = jacobian[:, dofs]
        normal = jacobian.T @ jacobian
        normal[np.diag_indices_from(normal)] += damping**2
        solution = np.linalg.solve(normal, jacobian.T @ error)
        if dofs is None:
            delta_q = solution
        else:
            delta_q = np.zeros(self.model.nv)
            delta_q[dofs] = solution
        mujoco.mj_integratePos(self.model, self.data.qpos, delta_q, 1.0)
        self._clamp()

    def _collision_rows(self, collision, weight, previous_arm=None, smooth_weight=0.0):
        """One-sided rows d(dist)/dq . dq = safe - dist on arm joints.

        While any pair is inside the activation band, arm joints are also pulled toward
        the previous frame so the arm does not snap around an obstacle.
        """
        rows, errors, active = [], [], False
        for body1, body2, position, normal, dist, safe in collision.query(self.data.qpos):
            mujoco.mj_jac(self.model, self.data, self.jac_pos, None, position, body2)
            mujoco.mj_jac(self.model, self.data, self.jac_other, None, position, body1)
            gain = weight * min(1.0, (safe + collision.ramp - dist) / collision.ramp)
            row = np.zeros(self.model.nv)
            row[self.arm_dofs] = (normal @ (self.jac_pos - self.jac_other))[self.arm_dofs]
            rows.append(gain * row)
            errors.append(gain * max(safe - dist, 0.0))
            active = active or dist < safe
        if rows and smooth_weight and previous_arm is not None:
            for dof_address, qpos_address, value in zip(
                self.arm_dofs, self.arm_qpos, previous_arm, strict=True
            ):
                row = np.zeros(self.model.nv)
                row[dof_address] = smooth_weight
                rows.append(row)
                errors.append(smooth_weight * (value - self.data.qpos[qpos_address]))
        return rows, errors, active

    def solve(
        self, target_pos, target_quat, posture_bias, iterations: int,
        collision=None, collision_weight=0.0, dofs=None, previous_arm=None, smooth_weight=0.0,
    ):
        """Levenberg-Marquardt IK; with ``collision`` the arm keeps a safe distance.

        A step that raises the weighted residual is undone and retried with more damping,
        which stops the near-singular (straight) arm from oscillating between iterations.
        ``dofs`` restricts the update to those DoFs (the arm pass keeps root and legs).
        Returns True if the collision constraint was active at the accepted solution.
        """
        damping = self.damping
        accepted = None
        for _ in range(iterations):
            jacobian, error = self._system(target_pos, target_quat, posture_bias)
            active = False
            if collision is not None:
                rows, errors, active = self._collision_rows(
                    collision, collision_weight, previous_arm, smooth_weight
                )
                if rows:
                    jacobian = np.vstack([jacobian, np.asarray(rows)])
                    error = np.concatenate([error, errors])
            cost = float(error @ error)
            if accepted is not None and cost > accepted[3]:
                self.data.qpos[:] = accepted[0]
                damping *= 4.0
            else:
                accepted = (self.data.qpos.copy(), jacobian, error, cost, active)
                damping = max(0.5 * damping, self.damping)
            self._step(accepted[1], accepted[2], damping, dofs)
        # Keep the last accepted iterate if the final trial step made things worse.
        jacobian, error = self._system(target_pos, target_quat, posture_bias)
        active = False
        if collision is not None:
            rows, errors, active = self._collision_rows(
                    collision, collision_weight, previous_arm, smooth_weight
                )
            if rows:
                error = np.concatenate([error, errors])
        if accepted is not None and float(error @ error) > accepted[3]:
            self.data.qpos[:] = accepted[0]
            active = accepted[4]
        return active

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
    arm_shift=0.0,
    shoulder_half_width=1.0,
    collision_options=None,
    arm_filter_sigma=0.0,
    progress=True,
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
    collision = None
    if collision_options is not None:
        checker = SelfCollision(
            model_file,
            {name: body_id for body_id, name in enumerate(body_names)},
            collision_options["body_dist"],
            collision_options["arm_dist"],
            collision_options["ramp"],
            collision_options["hand_inflate"],
        )
        collision = checker if collision_options["enabled"] else None
    hand_ids = [ik.body_ids[name] for name in _ARM_SHIFT_BODIES if name.endswith("hand_yaw_link")]
    hand_names = [name for name in _ARM_SHIFT_BODIES if name.endswith("hand_yaw_link")]
    frames = loader.output_frames
    hand_error = np.zeros(frames)
    arm_pass = bool(arm_shift) or collision is not None
    body_qpos = None
    previous_arm = None
    body_arm = np.zeros((frames, len(ik.arm_qpos)))
    frame_targets = []
    avoided = 0

    qpos = np.zeros((frames, model.nq))
    data.qpos[:] = model.qpos0
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    foot_error = 0.0
    frame_iter = range(frames)
    if progress:
        frame_iter = tqdm(frame_iter, desc=f"{output_file.stem} ik", leave=False)
    for frame in frame_iter:
        source_pos, source_quat = source.poses(
            loader.motion_base_poss[frame],
            loader.motion_base_rots[frame],
            loader.motion_dof_poss[frame],
        )
        body_targets = _build_targets(
            source_pos,
            source_quat,
            source.rest_quat,
            ik.rest_quat,
            horizontal_scale,
            vertical_scale,
            foot_offset,
        )
        bias = _posture_bias(ik.joints, list(loader.joint_names), loader.motion_dof_poss[frame])
        steps = warm_iterations if frame == 0 else iterations
        if frame == 0:
            data.qpos[:3] = body_targets[0][_ANCHOR_BODY]
            data.qpos[3:7] = body_targets[1][_ANCHOR_BODY]
        elif arm_pass:
            data.qpos[:] = body_qpos
        # Pass 1: whole-body IK exactly as without the arm handling (root, waist, legs).
        ik.solve(*body_targets, bias, steps)
        target_pos, target_quat = body_targets
        if arm_pass:
            body_qpos = data.qpos.copy()
            target_pos, target_quat = _build_targets(
                source_pos,
                source_quat,
                source.rest_quat,
                ik.rest_quat,
                horizontal_scale,
                vertical_scale,
                foot_offset,
                arm_shift,
                shoulder_half_width,
            )
            # Pass 2: arm joints only, with the width shift and collision avoidance.
            if previous_arm is not None:
                data.qpos[ik.arm_qpos] = previous_arm
            avoided += ik.solve(
                target_pos, target_quat, bias, steps, collision,
                collision_options["weight"] if collision is not None else 0.0,
                ik.arm_dofs, previous_arm,
                collision_options["smooth"] if collision is not None else 0.0,
            )
            previous_arm = data.qpos[ik.arm_qpos].copy()
            body_arm[frame] = body_qpos[ik.arm_qpos]
        frame_targets.append((target_pos, target_quat, bias))
        mujoco.mj_forward(model, data)
        qpos[frame] = data.qpos
        if debug:
            foot_error += sum(
                np.linalg.norm(target_pos[name] - data.xpos[ik.body_ids[name]])
                for name in _FOOT_BODIES
            )

    repaired = 0
    if arm_pass and arm_filter_sigma > 0 and frames > 1:
        # Low-pass only the arm correction on top of pass 1, keeping the source motion's
        # detail; then push any frame the filter left intersecting back out.
        correction = qpos[:, ik.arm_qpos] - body_arm
        qpos[:, ik.arm_qpos] = body_arm + _gaussian_filter(correction, arm_filter_sigma)
        if collision is not None:
            for frame in range(frames):
                if collision.max_penetration(qpos[frame]) <= 0.0:
                    continue
                data.qpos[:] = qpos[frame]
                # Stay near the filtered arm: a minimal push-out, not a fresh solve.
                ik.solve(
                    *frame_targets[frame], iterations, collision,
                    collision_options["weight"], ik.arm_dofs,
                    qpos[frame, ik.arm_qpos].copy(), _REPAIR_HOLD_WEIGHT,
                )
                qpos[frame] = data.qpos
                repaired += 1
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_kinematics(model, data)
        hand_error[frame] = max(
            np.linalg.norm(frame_targets[frame][0][name] - data.xpos[body_id])
            for name, body_id in zip(hand_names, hand_ids, strict=True)
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
    stats = {
        "frames": frames,
        "mean_foot_error": foot_error / max(2 * frames, 1),
        "grounded": lifted,
        "ground_penetration": -maximum_penetration,
        "avoided_frames": avoided,
        "repaired_frames": repaired,
        "mean_hand_error": float(hand_error.mean()),
        "max_hand_error": float(hand_error.max()),
    }
    arm_columns = [
        index
        for index, (name, _qa, _va, _joint_id) in enumerate(ik.joints)
        if any(key in name for key in _ARM_JOINT_KEYS)
    ]
    acceleration = np.diff(joint_pos[:, arm_columns], n=2, axis=0) * loader.output_fps**2
    stats["arm_acc_p99"] = float(np.percentile(np.abs(acceleration), 99)) if len(acceleration) else 0.0
    if collision_options is not None:
        depth = np.asarray([checker.max_penetration(qpos[frame]) for frame in range(frames)])
        longest = current = 0
        for value in depth >= _PENETRATION_REPORT:
            current = current + 1 if value else 0
            longest = max(longest, current)
        stats["max_self_penetration"] = float(depth.max(initial=0.0))
        stats["self_penetration_run_s"] = longest / loader.output_fps
    return stats


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
    parser.add_argument(
        "--arm_shift", type=float, default=-1.0,
        help="outward arm-target shift (m); -1=auto shoulder-width difference, 0=off",
    )
    parser.add_argument("--no_self_collision", action="store_true",
                        help="disable arm self-collision avoidance (penetration is still reported)")
    parser.add_argument("--collision_dist", type=float, default=0.015,
                        help="safe distance (m) between an arm and the trunk / legs")
    parser.add_argument("--collision_arm_dist", type=float, default=0.01,
                        help="safe distance (m) between the two arms")
    parser.add_argument("--collision_ramp", type=float, default=0.01,
                        help="band (m) beyond the safe distance where the constraint fades in")
    parser.add_argument("--collision_w", type=float, default=30.0)
    parser.add_argument("--collision_iters", type=int, default=20,
                        help="max extra arm-only IK iterations per frame in near-contact")
    parser.add_argument("--collision_smooth_w", type=float, default=0.3,
                        help="arm-joint pull toward the previous frame during avoidance")
    parser.add_argument("--arm_filter_sigma", type=float, default=1.5,
                        help="Gaussian sigma (frames) on the arm correction; 0=off")
    parser.add_argument("--hand_inflate", type=float, default=0.0,
                        help="grow the hand collision box half-sizes (m), e.g. for a larger real hand")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes, one clip per task (1=serial)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _convert_clip(csv_file, output_file, g1_model_xml, loader_kwargs, export_kwargs, progress):
    """Retarget one CSV clip (top-level so ProcessPoolExecutor workers can run it)."""
    loader = MotionLoader(motion_file=csv_file, **loader_kwargs)
    return run_export(
        loader,
        SourceFK(g1_model_xml, list(loader.joint_names)),
        output_file=output_file,
        progress=progress,
        **export_kwargs,
    )


_MANIFEST_FIELDS = (
    "name", "frames", "avoided_frames", "repaired_frames", "max_self_penetration", "self_penetration_run_s",
    "mean_hand_error", "max_hand_error", "arm_acc_p99", "mean_foot_error", "grounded",
    "ground_penetration",
)


def _write_manifest(output_dir: Path, rows: dict[str, dict]) -> Path:
    """Merge per-clip stats into <output_dir>/retarget_manifest.csv (re-runs stay complete)."""
    path = output_dir / "retarget_manifest.csv"
    merged = {}
    if path.exists():
        with path.open() as handle:
            merged = {row["name"]: row for row in csv.DictReader(handle)}
    merged.update(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for name in sorted(merged):
            writer.writerow(merged[name])
    return path


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
    auto_shift, shoulder_half_width = _arm_width_shift(
        args.g1_model_xml, args.gr2_model_xml, horizontal_scale
    )
    arm_shift = auto_shift if args.arm_shift < 0 else args.arm_shift
    print(f"[g1_csv_to_gr2_npz] {len(csv_files)} clip(s)")
    print(f"[g1_csv_to_gr2_npz] G1={args.g1_model_xml}")
    print(f"[g1_csv_to_gr2_npz] GR2={args.gr2_model_xml}")
    print(
        f"[g1_csv_to_gr2_npz] scales: h={horizontal_scale:.4f} "
        f"v={vertical_scale:.4f} foot={gr2_foot:.4f} arm_shift={arm_shift:.4f} "
        f"(G1 shoulder half width {shoulder_half_width:.4f})"
    )
    print(
        f"[g1_csv_to_gr2_npz] IK: iterations={args.ik_iters}, "
        f"warm={args.warm_iters}, damping={args.damping}, workers={args.workers}"
    )
    print(
        f"[g1_csv_to_gr2_npz] self-collision: "
        f"{'off' if args.no_self_collision else 'on'} dist={args.collision_dist} "
        f"arm_dist={args.collision_arm_dist} ramp={args.collision_ramp} "
        f"w={args.collision_w} iters={args.collision_iters} "
        f"smooth={args.collision_smooth_w} filter_sigma={args.arm_filter_sigma} "
        f"hand_inflate={args.hand_inflate}"
    )
    if args.dry_run:
        _validate_target(mujoco.MjModel.from_xml_path(args.gr2_model_xml))
        print(f"[g1_csv_to_gr2_npz] dry-run OK: {output_files[0]}")
        return

    loader_kwargs = dict(
        input_fps=int(args.input_fps),
        output_fps=int(args.output_fps),
        position_scale=args.position_scale,
        euler_order=args.euler_order,
    )
    export_kwargs = dict(
        model_file=args.gr2_model_xml,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        foot_offset=gr2_foot,
        damping=args.damping,
        posture_weight=args.posture_w,
        iterations=args.ik_iters,
        warm_iterations=args.warm_iters,
        debug=args.debug,
        ground=not args.no_ground,
        ground_tolerance=args.ground_tol,
        arm_shift=arm_shift,
        shoulder_half_width=shoulder_half_width,
        arm_filter_sigma=args.arm_filter_sigma,
        collision_options=dict(
            enabled=not args.no_self_collision,
            body_dist=args.collision_dist,
            arm_dist=args.collision_arm_dist,
            ramp=args.collision_ramp,
            weight=args.collision_w,
            iters=args.collision_iters,
            smooth=args.collision_smooth_w,
            hand_inflate=args.hand_inflate,
        ),
    )

    failures = []
    skipped = 0
    jobs = []
    for csv_file, output_file in zip(csv_files, output_files, strict=True):
        if args.skip_existing and output_file.exists():
            skipped += 1
        else:
            jobs.append((csv_file, output_file))

    rows = {}

    def report(output_file, stats):
        rows[output_file.stem] = {"name": output_file.stem, **stats}
        if args.debug:
            tqdm.write(
                f"[g1_csv_to_gr2_npz] {output_file.name}: "
                f"self_pen={stats['max_self_penetration']:.4f} m "
                f"({stats['self_penetration_run_s']:.2f} s >= {_PENETRATION_REPORT} m), "
                f"avoided={stats['avoided_frames']}/{stats['frames']}, "
                f"hand_err={stats['mean_hand_error']:.4f}/{stats['max_hand_error']:.4f} m, "
                f"foot_err={stats['mean_foot_error']:.4f} m, grounded={stats['grounded']}"
            )

    if args.workers <= 1:
        for csv_file, output_file in jobs:
            try:
                report(output_file, _convert_clip(
                    csv_file, output_file, args.g1_model_xml, loader_kwargs, export_kwargs, True
                ))
            except Exception as error:
                failures.append((csv_file, str(error)))
                print(f"[g1_csv_to_gr2_npz] FAILED {csv_file.name}: {error}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _convert_clip, csv_file, output_file, args.g1_model_xml,
                    loader_kwargs, export_kwargs, False,
                ): (csv_file, output_file)
                for csv_file, output_file in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="clips"):
                csv_file, output_file = futures[future]
                try:
                    report(output_file, future.result())
                except Exception as error:
                    failures.append((csv_file, str(error)))
                    tqdm.write(f"[g1_csv_to_gr2_npz] FAILED {csv_file.name}: {error}")

    if rows:
        manifest = _write_manifest(output_files[0].parent, rows)
        flagged = sum(
            1 for stats in rows.values() if stats["self_penetration_run_s"] >= 0.3
        )
        print(
            f"[g1_csv_to_gr2_npz] manifest: {manifest} "
            f"({flagged} clip(s) with >= {_PENETRATION_REPORT} m self-penetration for >= 0.3 s)"
        )
    converted = len(csv_files) - skipped - len(failures)
    print(f"[g1_csv_to_gr2_npz] done: {converted}/{len(csv_files)} converted")
    if skipped:
        print(f"[g1_csv_to_gr2_npz] {skipped} skipped (already existed)")
    if failures:
        for csv_file, error in failures:
            print(f"  {csv_file}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
