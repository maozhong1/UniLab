"""Convert G1 29-DoF CSV motions to H2 29-DoF tracking NPZ via task-space IK retarget.

Copying G1 joint angles into H2 by name (and rescaling only root-Z) does not work: H2's
legs are ~1.43x longer than G1's (and the shank grows more than the thigh — H2 shank/thigh
= 1.13 vs G1 0.94), so copied joint angles put H2 feet off the ground and drift the CoM. Here
we instead solve H2 joint angles so H2's *bodies* land where a size-scaled G1 pose puts them.

Pipeline (MuJoCo-only, same NPZ contract as the G1 ``bones_seed_csv_to_npz.py``):

1. ``MotionLoader`` parses the G1 CSV (29 ``*_joint_dof`` cols + root), resamples 120->50 fps
   and computes root/joint velocities (reused verbatim from the G1 converter).
2. **Source FK**: inject the G1 pose into a G1 MuJoCo model per frame and read the world
   pose of the 14 tracked bodies (the exact bodies the H2 env tracks / terminates on).
3. **Target scaling** (contact-preserving): build H2 body-position targets from the G1 poses
   with anisotropic scale — vertical is measured *up from the support foot* and scaled by the
   pelvis-height ratio (so a grounded foot stays grounded and pelvis lands near H2 nominal),
   horizontal is scaled by the leg-length ratio (stance width / stride / reach grow with the
   longer legs). Body orientations (pelvis, torso, feet) are scale-free and copied directly.
4. **IK**: a mink-style damped-least-squares solver (native ``mj_jacBody`` Jacobians, no extra
   dependency) drives H2 qpos to hit those targets each frame, warm-started from the previous
   frame for temporal continuity, with a posture-regularization task that softly pulls the
   *redundant* DoFs (arm yaws, head, etc.) toward the naive G1-name copy — hybrid, not naive.
   Foot orientation is matched on H2's *distal* foot ``ankle_pitch_link``: H2's ankle chain
   is roll->pitch, the reverse of G1's pitch->roll, so H2 ``ankle_roll_link`` is only an
   intermediate link (``--legacy_foot_ori`` restores the old, incorrect roll-link target).
5. **Velocity + readback**: qvel is recovered by ``mj_differentiatePos`` between solved frames
   (correct free-joint quaternion handling), then a final ``mj_forward`` pass reads the 29
   H2 hinge joints (model order) and per-body ``track_*`` tracking sensors.

Output NPZ keys match the tracking loader:
``fps, joint_pos(T,29), joint_vel(T,29), body_pos_w(T,nbody,3), body_quat_w(T,nbody,4),
body_lin_vel_w(T,nbody,3), body_ang_vel_w(T,nbody,3)``.

Validate a clip with ``scripts/motion/replay_npz.py`` before long training — check that feet
sit on the floor through the stance phase and the pelvis does not sink or float.

Usage:
    uv run scripts/motion/g1_csv_to_h2_npz2.py --input <csv_or_dir> --output <dir>
    uv run scripts/motion/g1_csv_to_h2_npz2.py --input clip.csv --output out.npz --debug
    uv run scripts/motion/g1_csv_to_h2_npz2.py --input <dir> --output <dir> --limit 20
    uv run scripts/motion/g1_csv_to_h2_npz2.py --input <dir> --output <dir> --workers 10
    uv run scripts/motion/g1_csv_to_h2_npz2.py --input <dir> --output <dir> \
        --align_sole_center --foot_forward_extra 0.025   # standing CoM near mid-foot
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One BLAS thread per process: the IK solves are tiny (35x35) and --workers parallelizes
# across clips, so multi-threaded BLAS would only oversubscribe the cores.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from unilab.assets import ASSETS_ROOT_PATH  # noqa: E402
from unilab.base.backend.mujoco.xml import inject_mujoco_tracking_sensors  # noqa: E402
from unilab.tools.bones_seed_csv import resolve_input_files  # noqa: E402

# Reuse the validated CSV parser / interpolator from the G1 converter (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bones_seed_csv_to_npz import MotionLoader  # noqa: E402

DEFAULT_G1_MODEL = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
DEFAULT_H2_MODEL = str(ASSETS_ROOT_PATH / "robots" / "h2" / "scene_flat.xml")

# The 14 bodies the H2 sonic env tracks (privileged_mf_hist) and terminates on. All names
# exist in both G1 and H2 (identical joint/body naming), so they are valid IK targets.
_TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
_FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
_ANCHOR_BODY = "pelvis"
_TORSO_BODY = "torso_link"

# Distal arm bodies (elbow, wrist) are re-targeted as a *shoulder-anchored* offset scaled by
# the arm-length ratio instead of the leg-derived h_scale/v_scale. H2's arm/leg proportions
# differ from G1's, so scaling the hand targets by the leg ratio distorts reach — which hurts
# arm-dominant clips (e.g. macarena). Keyed body -> its shoulder anchor (per side).
_ARM_SHOULDER: dict[str, str] = {
    "left_elbow_link": "left_shoulder_roll_link",
    "left_wrist_yaw_link": "left_shoulder_roll_link",
    "right_elbow_link": "right_shoulder_roll_link",
    "right_wrist_yaw_link": "right_shoulder_roll_link",
}

# Per-body IK position weights (higher = tracked harder). Feet + pelvis dominate because the
# env's strict terminators key on foot position and pelvis height/orientation.
_POS_WEIGHTS: dict[str, float] = {
    "pelvis": 6.0,
    "left_ankle_roll_link": 10.0, "right_ankle_roll_link": 10.0,
    "left_knee_link": 3.0, "right_knee_link": 3.0,
    "left_hip_roll_link": 2.0, "right_hip_roll_link": 2.0,
    "torso_link": 4.0,
    "left_wrist_yaw_link": 2.0, "right_wrist_yaw_link": 2.0,
    "left_elbow_link": 1.0, "right_elbow_link": 1.0,
    "left_shoulder_roll_link": 1.5, "right_shoulder_roll_link": 1.5,
}
# Bodies whose orientation we also match (scale-free): pelvis, torso, both feet.
_ORI_WEIGHTS: dict[str, float] = {
    "pelvis": 5.0, "torso_link": 3.0,
    "left_ankle_roll_link": 4.0, "right_ankle_roll_link": 4.0,
}
# H2 body that receives each G1 orientation target (keyed by the G1 body in _ORI_WEIGHTS).
# G1's distal foot is ankle_roll_link (pitch joint proximal, roll distal); H2 reverses the
# chain, so its distal foot — the one carrying the sole contact geom — is ankle_pitch_link.
# Matching the G1 foot orientation on H2's intermediate ankle_roll_link (legacy behaviour)
# forces the shank upright, removes stance knee bend, and tilts the sole by the copied
# ankle pitch.
_ORI_H2_BODY: dict[str, str] = {
    "pelvis": "pelvis", "torso_link": "torso_link",
    "left_ankle_roll_link": "left_ankle_pitch_link",
    "right_ankle_roll_link": "right_ankle_pitch_link",
}
_ORI_H2_BODY_LEGACY: dict[str, str] = {name: name for name in _ORI_WEIGHTS}
# Distal foot body (the one carrying the sole contact geoms) per robot, keyed by the tracked
# foot (IK position body). Used only by --align_sole_center to locate each sole centre.
_G1_SOLE_BODY: dict[str, str] = {name: name for name in _FOOT_BODIES}
_H2_SOLE_BODY: dict[str, str] = {
    "left_ankle_roll_link": "left_ankle_pitch_link",
    "right_ankle_roll_link": "right_ankle_pitch_link",
}


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
    """(qpos_adr, dof_adr) for each CSV joint, resolved by name in the model."""
    inj: list[tuple[int, int]] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"CSV joint '{name}' not found in model")
        inj.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))
    return inj


def _sensor_addr_table(model: mujoco.MjModel) -> np.ndarray:
    """Per-body sensordata addresses of the injected ``track_*`` sensors (-1 = absent)."""
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


def resolve_outputs(input_path: str, output: str, csv_files: list[Path]) -> list[Path]:
    in_root = Path(input_path).expanduser().resolve()
    if in_root.is_file():
        out = Path(output).expanduser().resolve()
        if out.suffix.lower() == ".npz":
            return [out]
        return [out / f"{in_root.stem}_h2.npz"]
    out_root = Path(output).expanduser().resolve()
    return [out_root / f"{f.stem}_h2.npz" for f in csv_files]


def _body_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body '{name}' not found in model")
        out[name] = bid
    return out


def _standing_geometry(model_file: str) -> tuple[float, float, float, float]:
    """Return (pelvis_z, hip_to_foot_len, foot_z, arm_len) at the nominal keyframe pose.

    Used only at startup (cold path) to derive vertical/horizontal/arm retarget scales and the
    foot-origin offset from the two robots' actual kinematics, not hard-coded magic numbers.
    ``foot_z`` is the ankle_roll body-origin height above the floor when standing (the offset
    that keeps a grounded foot's *body origin* at its natural height after retarget). ``arm_len``
    is the 3D shoulder_roll->wrist_yaw distance, used to scale distal arm targets by H2's own
    arm proportions rather than the leg ratio.
    """
    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = _body_ids(model, (
        "pelvis", "left_hip_roll_link", "left_ankle_roll_link",
        "left_shoulder_roll_link", "left_wrist_yaw_link",
    ))
    pelvis_z = float(data.xpos[ids["pelvis"], 2])
    foot_z = float(data.xpos[ids["left_ankle_roll_link"], 2])
    hip_to_foot = float(data.xpos[ids["left_hip_roll_link"], 2] - foot_z)
    arm_len = float(np.linalg.norm(
        data.xpos[ids["left_shoulder_roll_link"]] - data.xpos[ids["left_wrist_yaw_link"]]
    ))
    return pelvis_z, hip_to_foot, foot_z, arm_len


def _sole_center_x(model_file: str, sole_bodies: dict[str, str]) -> float:
    """Fore-aft sole-centre offset (m) of the foot contact geoms from the tracked ankle origin.

    Cold path only (startup). The IK position target is the ``ankle_roll_link`` body *origin*,
    but the sole sits differently around that origin on G1 vs H2 (H2's sole is shifted back),
    so matching origins leaves the H2 CoM near the toes. For each foot we take every contact
    geom (``contype != 0``) on the distal foot body, project its fore-aft extent (sphere/capsule
    radius included) into the distal-foot frame, and return the midpoint of the [heel, toe]
    range relative to the tracked ankle origin, averaged over both feet. Measured at qpos0.
    """
    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    centers: list[float] = []
    for foot, sole in sole_bodies.items():
        ids = _body_ids(model, (foot, sole))
        rot = data.xmat[ids[sole]].reshape(3, 3)  # distal-foot frame (x = forward)
        xs: list[float] = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != ids[sole] or model.geom_contype[g] == 0:
                continue
            gtype = model.geom_type[g]
            size = model.geom_size[g]
            gx = float((data.geom_xpos[g] - data.xpos[ids[foot]]) @ rot[:, 0])
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = float(size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                axis_x = float(data.geom_xmat[g].reshape(3, 3)[:, 2] @ rot[:, 0])
                half = abs(axis_x) * float(size[1]) + float(size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                gmat = data.geom_xmat[g].reshape(3, 3)
                half = float(np.abs(gmat.T @ rot[:, 0]) @ size[:3])
            else:
                raise ValueError(f"unsupported foot contact geom type {gtype} on '{sole}'")
            xs.extend((gx - half, gx + half))
        if not xs:
            raise ValueError(f"no foot contact geoms (contype!=0) on '{sole}' in {model_file}")
        centers.append(0.5 * (min(xs) + max(xs)))
    return float(np.mean(centers))


def _quat_error_world(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector omega s.t. applying omega rotates q_current onto q_target.

    Matches the frame of ``mj_jacBody``'s rotational Jacobian (global). Returns a 3-vector
    (axis * angle), ~2*vec(q_delta) for small errors.
    """
    q_conj = np.empty(4)
    mujoco.mju_negQuat(q_conj, q_current)
    q_delta = np.empty(4)
    mujoco.mju_mulQuat(q_delta, q_target, q_conj)  # world-frame delta: q_target = q_delta * q_current
    if q_delta[0] < 0.0:
        q_delta = -q_delta  # shortest arc
    vec = q_delta[1:4]
    sin_half = np.linalg.norm(vec)
    if sin_half < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, q_delta[0])
    return (angle / sin_half) * vec


class _SourceFK:
    """G1 forward kinematics: inject a CSV pose, read tracked-body world poses."""

    def __init__(self, g1_model_file: str, joint_names: list[str]):
        self.model = mujoco.MjModel.from_xml_path(g1_model_file)
        self.data = mujoco.MjData(self.model)
        self.inj = _map_csv_joints(self.model, joint_names)
        self.body_ids = _body_ids(self.model, _TRACKED_BODIES)

    def poses(self, root_pos, root_quat, dof_pos) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        d = self.data
        d.qpos[0:3] = root_pos
        d.qpos[3:7] = root_quat
        for j, (qa, _va) in enumerate(self.inj):
            d.qpos[qa] = dof_pos[j]
        mujoco.mj_forward(self.model, d)
        pos = {name: d.xpos[bid].copy() for name, bid in self.body_ids.items()}
        quat = {name: d.xquat[bid].copy() for name, bid in self.body_ids.items()}
        return pos, quat


def _build_targets(
    g1_pos: dict[str, np.ndarray],
    g1_quat: dict[str, np.ndarray],
    h_scale: float,
    v_scale: float,
    foot_offset: float,
    arm_scale: float,
    foot_forward: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Scale G1 body world poses into contact-preserving H2 targets.

    Vertical: height above the G1 support foot, scaled by pelvis-height ratio, then lifted by
    ``foot_offset`` (H2's nominal ankle-origin height) -> a grounded foot's body origin lands
    at its natural H2 height and the pelvis lands near H2 nominal. Horizontal: offset from the
    pelvis, scaled by leg-length ratio -> stance/stride/reach grow with the longer legs.
    Distal arm bodies (elbow, wrist) are then overwritten with a shoulder-anchored offset
    scaled by ``arm_scale`` (H2's arm-length ratio) so hand reach follows H2's arm proportions
    instead of the leg ratio. Orientation is copied unchanged (scale-free).
    ``foot_forward`` (m, 0 = off) then slides both foot position targets forward along the
    feet's shared horizontal heading (mean yaw of the two foot orientation targets) so H2's
    sole centre — not its ankle origin — lands near the scaled G1 sole centre (see
    ``--align_sole_center``). Using one shared direction (not each foot's own, toed-out yaw)
    keeps stance width and the left-right foot offset unchanged.
    """
    support_z = min(g1_pos[f][2] for f in _FOOT_BODIES)  # world Z of the planted foot
    pelvis_xy = g1_pos[_ANCHOR_BODY][:2]
    tgt_pos: dict[str, np.ndarray] = {}
    for name, p in g1_pos.items():
        # Keep the global pelvis path at G1 scale; grow only the pelvis-relative offset
        # (stance width / stride / reach) by the leg-length ratio. NOTE: writing the first
        # term as ``pelvis_xy`` (not ``pelvis_xy * h_scale``) is load-bearing — scaling it
        # too makes the pelvis terms cancel and uniformly magnifies the whole world path.
        xy = pelvis_xy + (p[:2] - pelvis_xy) * h_scale
        z = (p[2] - support_z) * v_scale + foot_offset
        tgt_pos[name] = np.array([xy[0], xy[1], z], dtype=np.float64)
    # Re-target distal arm bodies as a full-3D offset from their (already-scaled) shoulder
    # target, scaled by the arm-length ratio. The shoulder sits near the torso and keeps its
    # leg/pelvis-scaled target; only the arm's internal reach is rescaled.
    for name, shoulder in _ARM_SHOULDER.items():
        tgt_pos[name] = tgt_pos[shoulder] + (g1_pos[name] - g1_pos[shoulder]) * arm_scale
    tgt_quat = {name: g1_quat[name].copy() for name in _ORI_WEIGHTS}
    if foot_forward != 0.0:
        rot = np.empty(9)
        heading = np.zeros(2)
        for name in _FOOT_BODIES:
            mujoco.mju_quat2Mat(rot, tgt_quat[name])
            foot_dir = rot.reshape(3, 3)[:2, 0]  # foot x-axis, horizontal (yaw) component only
            norm = float(np.linalg.norm(foot_dir))
            if norm > 1e-6:
                heading += foot_dir / norm
        norm = float(np.linalg.norm(heading))
        if norm > 1e-6:  # feet facing opposite ways -> no well-defined forward, leave as-is
            for name in _FOOT_BODIES:
                tgt_pos[name][:2] += foot_forward * heading / norm
    return tgt_pos, tgt_quat


def _smooth_ground_lift(raw_lift: np.ndarray, window: int) -> np.ndarray:
    """Smooth the per-frame grounding lift into gentle ramps (no floor re-penetration).

    ``raw_lift[i] >= 0`` is the rigid vertical lift frame ``i`` needs so its deepest sole sits
    on the floor. Applying it per-frame makes a Z sawtooth whose finite-difference velocity is
    a train of spurious root vertical-velocity spikes in the exported motion. Here we (1) dilate
    (moving max) so an isolated spike becomes a short plateau, (2) box-average the plateau edges
    into ramps, then (3) clamp back to ``>= raw_lift`` so a smoothed frame never dips below the
    floor again. Lift-only (never negative), so genuine flight/hop phases are preserved.
    """
    n = raw_lift.size
    if window <= 1 or n == 0 or not np.any(raw_lift > 0.0):
        return raw_lift
    half = window // 2
    dil = np.empty_like(raw_lift)
    for i in range(n):
        dil[i] = raw_lift[max(0, i - half):min(n, i + half + 1)].max()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth = np.convolve(dil, kernel, mode="same")
    return np.maximum(smooth, raw_lift)


class _H2Ik:
    """Damped-least-squares task-space IK over the H2 free base + 29 hinges."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        damping: float,
        posture_w: float,
        ori_bodies: dict[str, str],
    ):
        self.model = model
        self.data = data
        self.damping = damping
        self.posture_w = posture_w
        self.body_ids = _body_ids(model, _TRACKED_BODIES)
        h2_ori_ids = _body_ids(model, tuple(ori_bodies[name] for name in _ORI_WEIGHTS))
        self.ori_body_ids = {name: h2_ori_ids[ori_bodies[name]] for name in _ORI_WEIGHTS}
        self.nv = model.nv
        # hinge (qpos_adr, dof_adr) in model order + joint range clamps for the 29 hinges
        self.hinges = _hinge_joint_readback(model)
        self.jnt_range = np.array(
            [model.jnt_range[jid] for jid in range(model.njnt)
             if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE],
            dtype=np.float64,
        )
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

    def _clamp_hinges(self) -> None:
        for k, (qa, _va) in enumerate(self.hinges):
            lo, hi = self.jnt_range[k]
            if lo < hi:  # 0/0 range means unlimited
                self.data.qpos[qa] = min(max(self.data.qpos[qa], lo), hi)

    def solve(
        self,
        tgt_pos: dict[str, np.ndarray],
        tgt_quat: dict[str, np.ndarray],
        posture_bias: np.ndarray,
        iters: int,
    ) -> None:
        model, data = self.model, self.data
        n_pos = len(_POS_WEIGHTS)
        n_ori = len(_ORI_WEIGHTS)
        n_post = len(self.hinges)
        rows = 3 * n_pos + 3 * n_ori + n_post
        J = np.zeros((rows, self.nv))
        e = np.zeros(rows)
        for _ in range(iters):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            r = 0
            for name, w in _POS_WEIGHTS.items():
                bid = self.body_ids[name]
                mujoco.mj_jacBody(model, data, self._jacp, self._jacr, bid)
                J[r:r + 3] = w * self._jacp
                e[r:r + 3] = w * (tgt_pos[name] - data.xpos[bid])
                r += 3
            for name, w in _ORI_WEIGHTS.items():
                bid = self.ori_body_ids[name]
                mujoco.mj_jacBody(model, data, self._jacp, self._jacr, bid)
                J[r:r + 3] = w * self._jacr
                e[r:r + 3] = w * _quat_error_world(tgt_quat[name], data.xquat[bid])
                r += 3
            # posture task: pull each hinge dof toward the naive G1-copy bias
            for k, (qa, va) in enumerate(self.hinges):
                J[r, va] = self.posture_w
                e[r] = self.posture_w * (posture_bias[k] - data.qpos[qa])
                r += 1
            # damped least squares: dq = (JtJ + lambda^2 I)^-1 Jt e
            JtJ = J.T @ J
            JtJ[np.diag_indices_from(JtJ)] += self.damping ** 2
            dq = np.linalg.solve(JtJ, J.T @ e)
            mujoco.mj_integratePos(model, data.qpos, dq, 1.0)
            self._clamp_hinges()


def run_h2_ik_export(
    loader: MotionLoader,
    source: _SourceFK,
    h2_model_file: str,
    output_file: Path,
    h_scale: float,
    v_scale: float,
    foot_offset: float,
    arm_scale: float,
    damping: float,
    posture_w: float,
    iters: int,
    warm_iters: int,
    debug: bool,
    ground: bool = True,
    ground_tol: float = 0.0,
    ground_smooth: int = 5,
    legacy_foot_ori: bool = False,
    foot_forward: float = 0.0,
    progress: bool = True,
) -> dict[str, float]:
    tmp_model_path, _, _ = inject_mujoco_tracking_sensors(h2_model_file)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_model_path)
    finally:
        Path(tmp_model_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)

    # Foot contact geoms (the condim mesh copies on ankle_*; contype!=0). Used by the
    # grounding pass to project any floor penetration out of each solved frame.
    foot_contact_geoms = {
        g for g in range(model.ngeom)
        if model.geom_contype[g] != 0
        and (bn := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]))
        and "ankle" in bn
    }

    inj = _map_csv_joints(model, list(loader.joint_names))       # 29 CSV joints -> H2 addrs
    readback = _hinge_joint_readback(model)                       # 29 H2 hinges, model order
    sensor_adrs = _sensor_addr_table(model)
    num_frames = loader.output_frames
    num_joints = len(readback)
    num_bodies = model.nbody
    ik = _H2Ik(
        model, data, damping=damping, posture_w=posture_w,
        ori_bodies=_ORI_H2_BODY_LEGACY if legacy_foot_ori else _ORI_H2_BODY,
    )

    # --- pass 1: solve H2 qpos per frame (warm-started from the previous solution) ---
    qpos_traj = np.zeros((num_frames, model.nq), dtype=np.float64)
    # seed pose = H2 nominal keyframe (qpos0) with naive G1-copied hinges as posture bias
    data.qpos[:] = model.qpos0
    err_accum = 0.0
    arm_err_accum = 0.0
    for i in tqdm(range(num_frames), desc=f"{output_file.stem} ik", leave=False,
                  disable=not progress):
        g1_pos, g1_quat = source.poses(
            loader.motion_base_poss[i], loader.motion_base_rots[i], loader.motion_dof_poss[i]
        )
        tgt_pos, tgt_quat = _build_targets(
            g1_pos, g1_quat, h_scale, v_scale, foot_offset, arm_scale, foot_forward
        )
        # naive G1-name copy = posture bias for redundant DoFs (0 for H2-only head joints)
        bias = np.zeros(num_joints)
        for j, (qa, _va) in enumerate(inj):
            for k, (qa2, _va2) in enumerate(readback):
                if qa2 == qa:
                    bias[k] = loader.motion_dof_poss[i, j]
                    break
        # warm-start the pelvis at the target so the first solve is not fighting a bad root
        if i == 0:
            data.qpos[0:3] = tgt_pos[_ANCHOR_BODY]
            data.qpos[3:7] = tgt_quat[_ANCHOR_BODY]
        ik.solve(tgt_pos, tgt_quat, bias, iters=warm_iters if i == 0 else iters)
        qpos_traj[i] = data.qpos.copy()
        if debug:
            err_accum += float(np.linalg.norm(tgt_pos["left_ankle_roll_link"] - data.xpos[ik.body_ids["left_ankle_roll_link"]]))
            arm_err_accum += 0.5 * (
                float(np.linalg.norm(tgt_pos["left_wrist_yaw_link"] - data.xpos[ik.body_ids["left_wrist_yaw_link"]]))
                + float(np.linalg.norm(tgt_pos["right_wrist_yaw_link"] - data.xpos[ik.body_ids["right_wrist_yaw_link"]]))
            )

    # --- grounding pass: project floor penetration out of each frame ---
    # The IK targets the ankle-*origin* height, not the sole contact point, so a pitched
    # foot (and IK residual) can bury the sole below z=0. Here we rigidly lift the whole
    # body (free-joint z) per frame by the deepest foot penetration so the lowest sole
    # sits on the floor. Lift-only (never drop) preserves genuine flight/hop phases.
    ground_stats = {"lifted": 0, "max_pen": 0.0}
    if ground and foot_contact_geoms:
        # Collect the per-frame required lift first, then smooth it over time before applying,
        # so isolated single-frame corrections become gentle ramps instead of a Z sawtooth
        # (otherwise mj_differentiatePos turns each lift delta into a spurious root
        # vertical-velocity spike in the exported motion).
        raw_lift = np.zeros(num_frames, dtype=np.float64)
        for i in range(num_frames):
            data.qpos[:] = qpos_traj[i]
            mujoco.mj_forward(model, data)  # runs collision detection -> data.contact
            pen = 0.0
            for c in range(data.ncon):
                con = data.contact[c]
                if con.geom1 in foot_contact_geoms or con.geom2 in foot_contact_geoms:
                    pen = min(pen, float(con.dist))  # most-negative = deepest penetration
            if pen < -ground_tol:
                raw_lift[i] = -pen  # rigid vertical lift so deepest sole -> z=0
                ground_stats["max_pen"] = min(ground_stats["max_pen"], pen)
        lift = _smooth_ground_lift(raw_lift, ground_smooth)
        qpos_traj[:, 2] += lift
        ground_stats["lifted"] = int(np.count_nonzero(lift > 1e-9))
        if debug:
            print(f"   grounding: lifted {ground_stats['lifted']}/{num_frames} frames, "
                  f"max penetration fixed = {-ground_stats['max_pen']*100:.1f} cm, "
                  f"smooth window = {ground_smooth}")

    # --- pass 2: qvel via mj_differentiatePos, then FK readback of joints + tracking sensors ---
    dt = 1.0 / float(loader.output_fps)
    qvel_traj = np.zeros((num_frames, model.nv), dtype=np.float64)
    for i in range(num_frames):
        a = qpos_traj[max(i - 1, 0)]
        b = qpos_traj[min(i + 1, num_frames - 1)]
        span = (min(i + 1, num_frames - 1) - max(i - 1, 0)) * dt
        if span > 0:
            mujoco.mj_differentiatePos(model, qvel_traj[i], span, a, b)

    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

    for i in range(num_frames):
        data.qpos[:] = qpos_traj[i]
        data.qvel[:] = qvel_traj[i]
        mujoco.mj_forward(model, data)
        for k, (qa, va) in enumerate(readback):
            joint_pos[i, k] = data.qpos[qa]
            joint_vel[i, k] = data.qvel[va]
        for body_id in range(num_bodies):
            pos_adr, quat_adr, lin_adr, ang_adr = sensor_adrs[body_id]
            body_pos_w[i, body_id] = (
                data.sensordata[pos_adr:pos_adr + _SENSOR_DIMS[0]] if pos_adr >= 0 else data.xpos[body_id]
            )
            body_quat_w[i, body_id] = (
                data.sensordata[quat_adr:quat_adr + _SENSOR_DIMS[1]] if quat_adr >= 0 else data.xquat[body_id]
            )
            if lin_adr >= 0:
                body_lin_vel_w[i, body_id] = data.sensordata[lin_adr:lin_adr + _SENSOR_DIMS[2]]
            if ang_adr >= 0:
                body_ang_vel_w[i, body_id] = data.sensordata[ang_adr:ang_adr + _SENSOR_DIMS[3]]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Write to a hidden temp name and rename, so an interrupted run never leaves a truncated
    # *.npz that --skip_existing or the training launcher's *.npz glob would pick up.
    partial_file = output_file.with_name(f".{output_file.name}.partial")
    with open(partial_file, "wb") as handle:
        np.savez(
            handle,
            fps=np.array([loader.output_fps], dtype=np.int32),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
        )
    os.replace(partial_file, output_file)
    return {
        "mean_foot_err": err_accum / max(num_frames, 1) if debug else float("nan"),
        "mean_arm_err": arm_err_accum / max(num_frames, 1) if debug else float("nan"),
    }


def _convert_clip(
    csv_file: Path,
    output_file: Path,
    g1_model_xml: str,
    h2_model_xml: str,
    loader_kwargs: dict,
    export_kwargs: dict,
    progress: bool,
) -> dict[str, float]:
    """Retarget one CSV clip (top-level so ProcessPoolExecutor workers can run it)."""
    loader = MotionLoader(motion_file=csv_file, **loader_kwargs)
    source = _SourceFK(g1_model_xml, list(loader.joint_names))
    return run_h2_ik_export(
        loader, source, h2_model_xml, output_file, progress=progress, **export_kwargs
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IK-retarget G1 CSV motions to H2 29-DoF NPZ")
    p.add_argument("--input", required=True, help="G1 CSV file or directory of CSV files")
    p.add_argument("--output", required=True, help="Output .npz (file input) or directory")
    p.add_argument("--g1_model_xml", default=DEFAULT_G1_MODEL, help="G1 scene_flat.xml (source FK)")
    p.add_argument("--h2_model_xml", default=DEFAULT_H2_MODEL, help="H2 scene_flat.xml (IK target)")
    p.add_argument("--input_fps", type=float, default=120.0)
    p.add_argument("--output_fps", type=float, default=50.0)
    p.add_argument("--position_scale", type=float, default=0.01, help="cm->m on root translate")
    p.add_argument("--euler_order", type=str, default="xyz")
    p.add_argument("--h_scale", type=float, default=0.0, help="horizontal scale (0=auto: leg-length ratio)")
    p.add_argument("--v_scale", type=float, default=0.0, help="vertical scale (0=auto: pelvis-height ratio)")
    p.add_argument("--arm_scale", type=float, default=0.0, help="distal-arm reach scale (0=auto: arm-length ratio)")
    p.add_argument("--ik_iters", type=int, default=12, help="DLS iterations per frame")
    p.add_argument("--warm_iters", type=int, default=60, help="DLS iterations for the first frame")
    p.add_argument("--damping", type=float, default=0.1, help="DLS damping (lambda)")
    p.add_argument("--posture_w", type=float, default=0.15, help="posture-regularization weight")
    p.add_argument("--no_ground", action="store_true", help="disable the floor-penetration grounding pass")
    p.add_argument("--ground_tol", type=float, default=0.0, help="allowed foot penetration (m) before lifting")
    p.add_argument("--ground_smooth", type=int, default=5,
                   help="temporal window (frames) to smooth the grounding lift (1=off, per-frame)")
    p.add_argument("--legacy_foot_ori", action="store_true",
                   help="match foot orientation on H2 ankle_roll_link (old, incorrect behaviour)")
    p.add_argument("--align_sole_center", action="store_true",
                   help="shift H2 foot targets forward so H2's sole centre (not its ankle origin) "
                        "matches the scaled G1 sole centre (puts the standing CoM nearer mid-foot)")
    p.add_argument("--foot_forward_extra", type=float, default=0.0,
                   help="extra forward foot-target shift (m) on top of --align_sole_center")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes, one clip per task (1=serial)")
    p.add_argument("--limit", type=int, default=0, help="convert only the first N clips (0=all)")
    p.add_argument(
        "--skip_existing", action="store_true",
        help="skip clips whose output .npz already exists (resume an interrupted batch run)",
    )
    p.add_argument("--dry-run", action="store_true", help="validate/plan without writing NPZ")
    p.add_argument("--debug", action="store_true", help="report mean foot-target residual per clip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_files = resolve_input_files(args.input)
    if args.limit and args.limit > 0:
        csv_files = csv_files[: args.limit]
    output_files = resolve_outputs(args.input, args.output, csv_files)

    # Derive retarget scales from the two robots' actual standing kinematics (cold path).
    g1_pelvis, g1_leg, _g1_foot, g1_arm = _standing_geometry(args.g1_model_xml)
    h2_pelvis, h2_leg, h2_foot, h2_arm = _standing_geometry(args.h2_model_xml)
    h_scale = args.h_scale or (h2_leg / g1_leg)
    v_scale = args.v_scale or (h2_pelvis / g1_pelvis)
    arm_scale = args.arm_scale or (h2_arm / g1_arm)
    foot_offset = h2_foot  # H2 nominal ankle-origin height -> grounded feet land naturally
    # Optional fore-aft sole alignment: delta = c_g1 * h_scale - c_h2 + extra (cold path).
    foot_forward = 0.0
    if args.align_sole_center:
        c_g1 = _sole_center_x(args.g1_model_xml, _G1_SOLE_BODY)
        c_h2 = _sole_center_x(args.h2_model_xml, _H2_SOLE_BODY)
        foot_forward = c_g1 * h_scale - c_h2 + args.foot_forward_extra
        print(f"[g1_csv_to_h2_npz2] sole centre: G1 {c_g1:+.4f} m, H2 {c_h2:+.4f} m -> "
              f"foot_forward={foot_forward:+.4f} m (extra {args.foot_forward_extra:+.4f})")
    elif args.foot_forward_extra:
        raise SystemExit("--foot_forward_extra requires --align_sole_center")

    print(f"[g1_csv_to_h2_npz2] {len(csv_files)} clip(s); G1={args.g1_model_xml}  H2={args.h2_model_xml}")
    print(f"[g1_csv_to_h2_npz2] input_fps={args.input_fps:g} output_fps={args.output_fps:g}")
    print(f"[g1_csv_to_h2_npz2] scales: h_scale={h_scale:.4f} (leg {g1_leg:.3f}->{h2_leg:.3f}) "
          f"v_scale={v_scale:.4f} (pelvis {g1_pelvis:.3f}->{h2_pelvis:.3f}) "
          f"arm_scale={arm_scale:.4f} (arm {g1_arm:.3f}->{h2_arm:.3f})")
    print(f"[g1_csv_to_h2_npz2] ik: iters={args.ik_iters} warm={args.warm_iters} "
          f"damping={args.damping} posture_w={args.posture_w}")
    if args.dry_run:
        print(f"[g1_csv_to_h2_npz2] dry-run OK. Example output: {output_files[0]}")
        return

    export_kwargs = dict(
        h_scale=h_scale, v_scale=v_scale, foot_offset=foot_offset, arm_scale=arm_scale,
        damping=args.damping, posture_w=args.posture_w, iters=args.ik_iters,
        warm_iters=args.warm_iters, debug=args.debug, ground=not args.no_ground,
        ground_tol=args.ground_tol, ground_smooth=args.ground_smooth,
        legacy_foot_ori=args.legacy_foot_ori, foot_forward=foot_forward,
    )
    loader_kwargs = dict(
        input_fps=int(args.input_fps), output_fps=int(args.output_fps),
        position_scale=args.position_scale, euler_order=args.euler_order,
    )
    print(f"[g1_csv_to_h2_npz2] foot orientation target: "
          f"{'ankle_roll_link (legacy)' if args.legacy_foot_ori else 'ankle_pitch_link'}; "
          f"workers={args.workers}")

    failures: list[tuple[Path, str]] = []
    skipped = 0
    jobs: list[tuple[Path, Path]] = []
    for csv_file, output_file in zip(csv_files, output_files, strict=True):
        if args.skip_existing and output_file.exists():
            skipped += 1
        else:
            jobs.append((csv_file, output_file))

    def report(output_file: Path, stats: dict[str, float]) -> None:
        if args.debug:
            tqdm.write(f"[g1_csv_to_h2_npz2] {output_file.name}: "
                       f"mean_foot_err={stats['mean_foot_err']:.4f} m  "
                       f"mean_arm_err={stats['mean_arm_err']:.4f} m")

    if args.workers <= 1:
        for csv_file, output_file in jobs:
            try:
                report(output_file, _convert_clip(
                    csv_file, output_file, args.g1_model_xml, args.h2_model_xml,
                    loader_kwargs, export_kwargs, True,
                ))
            except Exception as exc:  # keep going on a bad clip
                failures.append((csv_file, str(exc)))
                print(f"[g1_csv_to_h2_npz2] FAILED {csv_file.name}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _convert_clip, csv_file, output_file, args.g1_model_xml,
                    args.h2_model_xml, loader_kwargs, export_kwargs, False,
                ): (csv_file, output_file)
                for csv_file, output_file in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="clips"):
                csv_file, output_file = futures[future]
                try:
                    report(output_file, future.result())
                except Exception as exc:  # keep going on a bad clip
                    failures.append((csv_file, str(exc)))
                    tqdm.write(f"[g1_csv_to_h2_npz2] FAILED {csv_file.name}: {exc}")

    ok = len(csv_files) - len(failures) - skipped
    print(f"[g1_csv_to_h2_npz2] done: {ok}/{len(csv_files)} converted -> {output_files[0].parent}")
    if skipped:
        print(f"[g1_csv_to_h2_npz2] {skipped} skipped (already existed)")
    if failures:
        print(f"[g1_csv_to_h2_npz2] {len(failures)} failed:")
        for csv_file, message in failures:
            print(f"    {csv_file}: {message}")


if __name__ == "__main__":
    main()
