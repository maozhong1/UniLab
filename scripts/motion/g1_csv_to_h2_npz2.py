"""Convert G1 29-DoF CSV motions to H2 31-DoF tracking NPZ via task-space IK retarget.

This is the *proper retarget* successor to the naive ``g1_csv_to_h2_npz.py``. The naive
converter injects G1 joint angles into H2 by name and rescales only root-Z; because H2's
legs are ~1.43x longer than G1's (and the shank grows more than the thigh — H2 shank/thigh
= 1.13 vs G1 0.94), copied joint angles put H2 feet off the ground and drift the CoM. Here
we instead solve H2 joint angles so H2's *bodies* land where a size-scaled G1 pose puts them.

Pipeline (MuJoCo-only, same NPZ contract as the naive converter):

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
5. **Velocity + readback**: qvel is recovered by ``mj_differentiatePos`` between solved frames
   (correct free-joint quaternion handling), then a final ``mj_forward`` pass reads the 31
   H2 hinge joints (model order) and per-body tracking sensors exactly like the naive path.

Output NPZ keys are identical to the naive converter and the tracking loader:
``fps, joint_pos(T,31), joint_vel(T,31), body_pos_w(T,nbody,3), body_quat_w(T,nbody,4),
body_lin_vel_w(T,nbody,3), body_ang_vel_w(T,nbody,3)``.

Validate a clip with ``scripts/motion/replay_npz.py`` before long training — check that feet
sit on the floor through the stance phase and the pelvis does not sink or float.

Usage:
    uv run scripts/motion/g1_csv_to_h2_npz2.py --input <csv_or_dir> --output <dir>
    uv run scripts/motion/g1_csv_to_h2_npz2.py --input clip.csv --output out.npz --debug
    uv run scripts/motion/g1_csv_to_h2_npz2.py --input <dir> --output <dir> --limit 20
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

# Reuse the validated CSV parser and the shared readback/sensor helpers from the naive
# converter (same dir) instead of re-implementing them here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_csv_to_h2_npz import (  # noqa: E402, I001
    MotionLoader,
    _hinge_joint_readback,
    _map_csv_joints,
    _sensor_addr_table,
    _SENSOR_DIMS,
    resolve_input_files,
    resolve_outputs,
)

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


def _body_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body '{name}' not found in model")
        out[name] = bid
    return out


def _standing_geometry(model_file: str) -> tuple[float, float, float]:
    """Return (pelvis_z, hip_to_foot_len, foot_z) for a model at its nominal keyframe pose.

    Used only at startup (cold path) to derive vertical/horizontal retarget scales and the
    foot-origin offset from the two robots' actual kinematics, not hard-coded magic numbers.
    ``foot_z`` is the ankle_roll body-origin height above the floor when standing (the offset
    that keeps a grounded foot's *body origin* at its natural height after retarget).
    """
    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = _body_ids(model, ("pelvis", "left_hip_roll_link", "left_ankle_roll_link"))
    pelvis_z = float(data.xpos[ids["pelvis"], 2])
    foot_z = float(data.xpos[ids["left_ankle_roll_link"], 2])
    hip_to_foot = float(data.xpos[ids["left_hip_roll_link"], 2] - foot_z)
    return pelvis_z, hip_to_foot, foot_z


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
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Scale G1 body world poses into contact-preserving H2 targets.

    Vertical: height above the G1 support foot, scaled by pelvis-height ratio, then lifted by
    ``foot_offset`` (H2's nominal ankle-origin height) -> a grounded foot's body origin lands
    at its natural H2 height and the pelvis lands near H2 nominal. Horizontal: offset from the
    pelvis, scaled by leg-length ratio -> stance/stride/reach grow with the longer legs.
    Orientation is copied unchanged (scale-free).
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
    tgt_quat = {name: g1_quat[name].copy() for name in _ORI_WEIGHTS}
    return tgt_pos, tgt_quat


class _H2Ik:
    """Damped-least-squares task-space IK over the H2 free base + 31 hinges."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, damping: float, posture_w: float):
        self.model = model
        self.data = data
        self.damping = damping
        self.posture_w = posture_w
        self.body_ids = _body_ids(model, _TRACKED_BODIES)
        self.nv = model.nv
        # hinge (qpos_adr, dof_adr) in model order + joint range clamps for the 31 hinges
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
                bid = self.body_ids[name]
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
    damping: float,
    posture_w: float,
    iters: int,
    warm_iters: int,
    debug: bool,
    ground: bool = True,
    ground_tol: float = 0.0,
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
    readback = _hinge_joint_readback(model)                       # 31 H2 hinges, model order
    sensor_adrs = _sensor_addr_table(model)
    num_frames = loader.output_frames
    num_joints = len(readback)
    num_bodies = model.nbody
    ik = _H2Ik(model, data, damping=damping, posture_w=posture_w)

    # --- pass 1: solve H2 qpos per frame (warm-started from the previous solution) ---
    qpos_traj = np.zeros((num_frames, model.nq), dtype=np.float64)
    # seed pose = H2 nominal keyframe (qpos0) with naive G1-copied hinges as posture bias
    data.qpos[:] = model.qpos0
    err_accum = 0.0
    for i in tqdm(range(num_frames), desc=f"{output_file.stem} ik", leave=False):
        g1_pos, g1_quat = source.poses(
            loader.motion_base_poss[i], loader.motion_base_rots[i], loader.motion_dof_poss[i]
        )
        tgt_pos, tgt_quat = _build_targets(g1_pos, g1_quat, h_scale, v_scale, foot_offset)
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

    # --- grounding pass: project floor penetration out of each frame ---
    # The IK targets the ankle-*origin* height, not the sole contact point, so a pitched
    # foot (and IK residual) can bury the sole below z=0. Here we rigidly lift the whole
    # body (free-joint z) per frame by the deepest foot penetration so the lowest sole
    # sits on the floor. Lift-only (never drop) preserves genuine flight/hop phases.
    ground_stats = {"lifted": 0, "max_pen": 0.0}
    if ground and foot_contact_geoms:
        for i in range(num_frames):
            data.qpos[:] = qpos_traj[i]
            mujoco.mj_forward(model, data)  # runs collision detection -> data.contact
            pen = 0.0
            for c in range(data.ncon):
                con = data.contact[c]
                if con.geom1 in foot_contact_geoms or con.geom2 in foot_contact_geoms:
                    pen = min(pen, float(con.dist))  # most-negative = deepest penetration
            if pen < -ground_tol:
                qpos_traj[i, 2] += -pen  # rigid vertical lift so deepest sole -> z=0
                ground_stats["lifted"] += 1
                ground_stats["max_pen"] = min(ground_stats["max_pen"], pen)
        if debug:
            print(f"   grounding: lifted {ground_stats['lifted']}/{num_frames} frames, "
                  f"max penetration fixed = {-ground_stats['max_pen']*100:.1f} cm")

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
    np.savez(
        output_file,
        fps=np.array([loader.output_fps], dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    return {"mean_foot_err": err_accum / max(num_frames, 1) if debug else float("nan")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IK-retarget G1 CSV motions to H2 31-DoF NPZ")
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
    p.add_argument("--ik_iters", type=int, default=12, help="DLS iterations per frame")
    p.add_argument("--warm_iters", type=int, default=60, help="DLS iterations for the first frame")
    p.add_argument("--damping", type=float, default=0.1, help="DLS damping (lambda)")
    p.add_argument("--posture_w", type=float, default=0.15, help="posture-regularization weight")
    p.add_argument("--no_ground", action="store_true", help="disable the floor-penetration grounding pass")
    p.add_argument("--ground_tol", type=float, default=0.0, help="allowed foot penetration (m) before lifting")
    p.add_argument("--limit", type=int, default=0, help="convert only the first N clips (0=all)")
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
    g1_pelvis, g1_leg, _g1_foot = _standing_geometry(args.g1_model_xml)
    h2_pelvis, h2_leg, h2_foot = _standing_geometry(args.h2_model_xml)
    h_scale = args.h_scale or (h2_leg / g1_leg)
    v_scale = args.v_scale or (h2_pelvis / g1_pelvis)
    foot_offset = h2_foot  # H2 nominal ankle-origin height -> grounded feet land naturally

    print(f"[g1_csv_to_h2_npz2] {len(csv_files)} clip(s); G1={args.g1_model_xml}  H2={args.h2_model_xml}")
    print(f"[g1_csv_to_h2_npz2] input_fps={args.input_fps:g} output_fps={args.output_fps:g}")
    print(f"[g1_csv_to_h2_npz2] scales: h_scale={h_scale:.4f} (leg {g1_leg:.3f}->{h2_leg:.3f}) "
          f"v_scale={v_scale:.4f} (pelvis {g1_pelvis:.3f}->{h2_pelvis:.3f})")
    print(f"[g1_csv_to_h2_npz2] ik: iters={args.ik_iters} warm={args.warm_iters} "
          f"damping={args.damping} posture_w={args.posture_w}")
    if args.dry_run:
        print(f"[g1_csv_to_h2_npz2] dry-run OK. Example output: {output_files[0]}")
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
            source = _SourceFK(args.g1_model_xml, list(loader.joint_names))
            stats = run_h2_ik_export(
                loader, source, args.h2_model_xml, output_file,
                h_scale=h_scale, v_scale=v_scale, foot_offset=foot_offset, damping=args.damping,
                posture_w=args.posture_w, iters=args.ik_iters, warm_iters=args.warm_iters,
                debug=args.debug, ground=not args.no_ground, ground_tol=args.ground_tol,
            )
            if args.debug:
                print(f"[g1_csv_to_h2_npz2] {output_file.name}: mean_foot_err={stats['mean_foot_err']:.4f} m")
        except Exception as exc:  # keep going on a bad clip
            failures.append((csv_file, str(exc)))
            print(f"[g1_csv_to_h2_npz2] FAILED {csv_file.name}: {exc}")

    ok = len(csv_files) - len(failures)
    print(f"[g1_csv_to_h2_npz2] done: {ok}/{len(csv_files)} converted -> {output_files[0].parent}")
    if failures:
        print(f"[g1_csv_to_h2_npz2] {len(failures)} failed")


if __name__ == "__main__":
    main()
