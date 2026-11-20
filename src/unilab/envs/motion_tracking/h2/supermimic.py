"""Unitree H2 (fixed head, 29 DoF) whole-body tracking env with SuperMimic semantics.

Data-collection env of the SuperMimic world-model + BPTT algorithm
(``unilab.algos.torch.supermimic``). It reproduces the mjlab env of the SuperMimic reference
implementation (``supermimic/envs/tracking_env_cfg.py``) on UniLab's NumPy backends:

* observation ``obs`` (10 x (3N+6) = 930): ``q + encoder_bias - q0``, ``qd``, root angular
  velocity (body frame), projected gravity, last clean action; uniform sensor noise, 10-frame
  history flattened **term-major** ``[q_t0..q_t9 | qd_t0..qd_t9 | ...]`` (oldest first);
* observation ``goal`` (10 x (2N+10) = 680): noise-free features of the next 10 reference
  frames, clamped at the clip end;
* action: ``target = q0 + (a + eps) * scale - encoder_bias`` with ``eps ~ N(0, 0.1^2)``
  (execution noise; observation and action-rate penalty see the clean action);
* reward: BeyondMimic tracking terms (root / re-anchored body pose, body velocities), action
  rate and soft joint limits, multiplied by ``ctrl_dt``;
* terminations: pelvis height error > 0.25 m or gravity-tilt error > 0.8 versus the reference;
  10 s time-out and clip end are truncations;
* reset: reference-state initialisation at a frame chosen by return-based bin sampling, with
  pose / velocity / joint noise;
* domain randomisation (sim context ``xi`` = friction 1 + encoder bias N + torso COM offset 3):
  foot friction and encoder bias per episode, torso COM offset fixed per env.

Target-frame timing follows mjlab: after a reset to frame ``f`` the target is ``f + 1``; the
reward of a step is computed against the target frame the step was taken under, then the
target advances by one frame.

The algorithm reads the per-env simulator state ``s = [qpos | qvel]`` (13+2N), the context,
the executed actions and the target frames through the ``sm_*`` attributes below; the
``obs`` / ``goal`` groups are what the deployed policy consumes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dr.types import (
    RESET_TERM_BODY_IPOS,
    RESET_TERM_GEOM_FRICTION,
    ResetRandomizationPayload,
)
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.utils.rotation import (
    np_quat_apply_batched,
    np_quat_from_euler_xyz,
    np_quat_mul,
    np_quat_mul_batched,
    np_yaw_quat,
)

from .supermimic_library import (
    ReturnBasedSampler,
    ReturnSamplerCfg,
    SuperMimicMotionLibrary,
    discover_motion_files,
    quat_apply_inverse,
)

H2_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# The 14 G1 / UniLab SONIC bodies plus both distal foot bodies. H2's ankle chain is roll -> pitch,
# so ``ankle_roll_link`` is proximal and only ``ankle_pitch_link`` carries the sole orientation.
H2_TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
)

# Observation term layout (term-major history order) and uniform noise half-widths.
OBS_TERMS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "base_ang_vel",
    "projected_gravity",
    "actions",
)

_GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


@dataclass
class SuperMimicRewardCfg:
    """Tracking reward (SuperMimic Table A.2); every term is multiplied by ``ctrl_dt``."""

    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0

    @property
    def max_weight_sum(self) -> float:
        return (
            self.w_root_pos
            + self.w_root_ori
            + self.w_body_pos
            + self.w_body_ori
            + self.w_body_lin_vel
            + self.w_body_ang_vel
        )


@dataclass
class SuperMimicDomainRandCfg:
    enabled: bool = True
    friction_range: tuple[float, float] = (0.3, 1.2)
    eval_friction: float = 1.0  # foot friction when ``enabled`` is False (evaluation protocol)
    encoder_bias_range: tuple[float, float] = (-0.01, 0.01)
    com_offset_range: tuple[tuple[float, float], ...] = (
        (-0.025, 0.025),
        (-0.05, 0.05),
        (-0.05, 0.05),
    )


@dataclass
class SuperMimicResetNoiseCfg:
    enabled: bool = True
    pose: tuple[tuple[float, float], ...] = (
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.01, 0.01),
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-0.2, 0.2),
    )
    velocity: tuple[tuple[float, float], ...] = (
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.2, 0.2),
        (-0.52, 0.52),
        (-0.52, 0.52),
        (-0.78, 0.78),
    )
    joint_pos: tuple[float, float] = (-0.1, 0.1)


@dataclass
class _Asset:
    base_name = "pelvis"
    foot_name = "ankle_pitch_link"
    ground = "floor"


@registry.envcfg("H2SuperMimic")
@dataclass
class H2SuperMimicCfg(G1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "h2" / "scene_flat_supermimic.xml")
        )
    )
    asset: _Asset = field(default_factory=_Asset)  # type: ignore[assignment]
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    max_episode_seconds: float | None = 10.0
    # Simulate a copy of the scene without visual meshes / skybox whose collision meshes keep only
    # their convex-hull vertices (identical contacts). Per-env friction / COM randomisation makes
    # the backend hold one model per env, and the full H2 model is 187 MB (2 MB stripped).
    strip_visual_meshes: bool = True

    # Reference motions: explicit npz files and/or directories of npz files (BONES-SEED H2 retarget).
    motion_file: str | list[str] = field(default_factory=list)
    motion_dirs: list[str] = field(default_factory=list)
    motion_manifest: str | None = None  # csv name,keep
    motion_limit: int | None = None
    # Clips whose name contains one of these keywords are dropped (prop interaction).
    motion_exclude_keywords: list[str] = field(default_factory=lambda: ["table", "_box_"])

    joint_names: tuple[str, ...] = H2_JOINT_NAMES
    tracked_bodies: tuple[str, ...] = H2_TRACKED_BODIES
    anchor_body_name: str = "pelvis"
    root_body_name: str = "pelvis"
    foot_geom_pattern: str = r"^(left|right)_foot[1-4]_collision$"
    com_body_name: str = "torso_link"

    history_length: int = 10
    future_frames: int = 10
    obs_noise: bool = True
    obs_noise_half: tuple[float, ...] = (0.01, 0.5, 0.2, 0.05, 0.0)  # per OBS_TERMS
    action_noise_std: float = 0.1
    # Action scale = 0.25 * effort / kp per actuator (effort = min(forcerange, actuatorfrcrange)).
    action_scale_factor: float = 0.25
    soft_joint_limit_factor: float = 0.9

    root_height_threshold: float = 0.25
    root_tilt_threshold: float = 0.8

    sampling_mode: str = "return"  # return | uniform | start
    bin_frames: int = 50
    sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)
    reset_noise: SuperMimicResetNoiseCfg = field(default_factory=SuperMimicResetNoiseCfg)
    domain_rand: SuperMimicDomainRandCfg = field(default_factory=SuperMimicDomainRandCfg)  # type: ignore[assignment]
    reward: SuperMimicRewardCfg = field(default_factory=SuperMimicRewardCfg)
    seed: int | None = None


def compute_action_scale(model: Any, factor: float = 0.25) -> np.ndarray:
    """``factor * effort / kp`` per position actuator, effort = min(actuator forcerange, joint actuatorfrcrange).

    The joint-level limit is what clamps the torque in MuJoCo (ankle roll: 150 vs 19), so the
    scale follows UniLab's ``tracking_sonic.py`` and SuperMimic's ``robots/h2.py``.
    """
    scale = np.zeros(model.nu)
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        effort = float(model.actuator_forcerange[i, 1])
        if model.jnt_actfrclimited[jid]:
            effort = min(effort, float(model.jnt_actfrcrange[jid, 1]))
        scale[i] = factor * effort / float(model.actuator_gainprm[i, 0])
    return scale


def collision_only_scene(model_file: str) -> str:
    """Write (once, cached by content) a physics-equivalent copy of ``model_file`` and return its path.

    Visual-only mesh geoms (``contype = conaffinity = 0``) and the skybox texture are dropped and
    every collision mesh is replaced by its convex-hull vertices, expressed in the compiled geom
    frame. MuJoCo collides meshes through their convex hull and the H2 bodies carry explicit
    inertials, so dynamics and contacts are unchanged.
    """
    import mujoco

    src = Path(model_file)
    digest = hashlib.sha1(
        f"{src.resolve()}:{src.stat().st_mtime_ns}:{mujoco.__version__}".encode()
    ).hexdigest()[:16]
    out = Path(tempfile.gettempdir()) / "unilab_collision_scenes" / f"{src.stem}_{digest}.xml"
    if out.exists():
        return str(out)
    model = mujoco.MjModel.from_xml_path(str(src))
    spec = mujoco.MjSpec.from_file(str(src))
    geoms = list(spec.geoms)
    if len(geoms) != model.ngeom:
        raise RuntimeError("spec / model geom count mismatch")
    for i, g in enumerate(geoms):
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if g.contype == 0 and g.conaffinity == 0:
            spec.delete(g)
            continue
        g.pos = model.geom_pos[i].copy()
        g.quat = model.geom_quat[i].copy()
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            spec.delete(mesh)
            continue
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh.name)
        adr = model.mesh_graphadr[mid]
        nv = model.mesh_graph[adr]
        hull = model.mesh_vert[model.mesh_vertadr[mid] + model.mesh_graph[adr + 2 : adr + 2 + nv]]
        mesh.file = ""
        mesh.scale = [1.0, 1.0, 1.0]
        mesh.uservert = hull.astype(np.float64).reshape(-1)
    for tex in list(spec.textures):
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(tex)
    spec.compile()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(spec.to_xml())
    tmp.replace(out)
    return str(out)


# ---------------------------------------------------------------------- reward


def _sq_quat_err(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Squared geodesic angle between unit quaternions (..., 4)."""
    rel = np_quat_mul_batched(q1, q2 * np.array([1.0, -1.0, -1.0, -1.0], dtype=q2.dtype))
    w = np.abs(rel[..., 0])
    s = np.linalg.norm(rel[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(s, w)
    return angle * angle


def reanchor_reference(
    ref_body_pos_w: np.ndarray,
    ref_body_quat_w: np.ndarray,
    ref_anchor_pos_w: np.ndarray,
    ref_anchor_quat_w: np.ndarray,
    robot_anchor_pos_w: np.ndarray,
    robot_anchor_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference bodies in a frame sharing the robot anchor's xy position and yaw (BeyondMimic)."""
    delta_pos = np.concatenate([robot_anchor_pos_w[:, :2], ref_anchor_pos_w[:, 2:3]], axis=-1)
    inv_ref = ref_anchor_quat_w * np.array([1.0, -1.0, -1.0, -1.0], dtype=ref_anchor_quat_w.dtype)
    delta_ori = np_yaw_quat(np_quat_mul(robot_anchor_quat_w, inv_ref)).astype(ref_body_quat_w.dtype)
    rel_quat = np_quat_mul_batched(delta_ori[:, None, :], ref_body_quat_w)
    rel_pos = delta_pos[:, None, :] + np_quat_apply_batched(
        delta_ori[:, None, :], ref_body_pos_w - ref_anchor_pos_w[:, None, :]
    )
    return rel_pos, rel_quat


def tracking_reward_terms(
    *,
    robot_body_pos_w: np.ndarray,
    robot_body_quat_w: np.ndarray,
    robot_body_lin_vel_w: np.ndarray,
    robot_body_ang_vel_w: np.ndarray,
    ref_body_pos_w: np.ndarray,
    ref_body_quat_w: np.ndarray,
    ref_body_lin_vel_w: np.ndarray,
    ref_body_ang_vel_w: np.ndarray,
    anchor_index: int,
    action: np.ndarray,
    prev_action: np.ndarray,
    joint_pos: np.ndarray,
    soft_limits: np.ndarray,
    cfg: SuperMimicRewardCfg,
    dt: float,
) -> dict[str, np.ndarray]:
    """Weighted, dt-scaled reward terms (n,) plus ``total``; NumPy twin of the algorithm's torch reward."""
    ra_pos, ra_quat = robot_body_pos_w[:, anchor_index], robot_body_quat_w[:, anchor_index]
    ma_pos, ma_quat = ref_body_pos_w[:, anchor_index], ref_body_quat_w[:, anchor_index]
    rel_pos, rel_quat = reanchor_reference(
        ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat
    )

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)
    lo, hi = soft_limits[:, 0], soft_limits[:, 1]
    joint_limit = (np.maximum(lo - joint_pos, 0.0) + np.maximum(joint_pos - hi, 0.0)).sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    terms = {
        "motion_global_root_pos": cfg.w_root_pos * np.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": cfg.w_root_ori * np.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": cfg.w_body_pos * np.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": cfg.w_body_ori * np.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": cfg.w_body_lin_vel * np.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": cfg.w_body_ang_vel * np.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": cfg.w_action_rate * action_rate,
        "joint_limit": cfg.w_joint_limit * joint_limit,
    }
    terms = {k: (dt * v).astype(np.float32) for k, v in terms.items()}
    terms["total"] = np.sum(np.stack(list(terms.values())), axis=0)
    return terms


# ---------------------------------------------------------------------- DR / reset provider


class SuperMimicResetProvider(DomainRandomizationProvider):
    """Reference-state initialisation plus per-episode friction / encoder bias, per-env COM."""

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        for term in (RESET_TERM_GEOM_FRICTION, RESET_TERM_BODY_IPOS):
            if not capabilities.supports_reset_term(term):
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support '{term}' reset randomization"
                )

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return None

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        return env._build_reset_plan(env_ids)

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return env._reset_observation(env_ids)


# ---------------------------------------------------------------------- env


@registry.env("H2SuperMimic", sim_backend="mujoco")
class H2SuperMimicEnv(G1BaseEnv):
    """H2 tracking env for SuperMimic data collection (see module docstring)."""

    _cfg: H2SuperMimicCfg

    def __init__(self, cfg: H2SuperMimicCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        scene = cfg.scene
        if cfg.strip_visual_meshes:
            scene = dataclasses.replace(scene, model_file=collision_only_scene(scene.model_file))
        backend = create_backend(
            backend_type,
            scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            add_body_sensors=True,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)
        self._rng = np.random.default_rng(cfg.seed)
        n, nj = num_envs, self._num_action
        if nj != len(cfg.joint_names):
            raise ValueError(f"model has {nj} actuators, cfg lists {len(cfg.joint_names)} joints")
        self.num_joints = nj
        self.state_dim = 13 + 2 * nj
        self.obs_frame_dim = 3 * nj + 6
        self.goal_frame_dim = 2 * nj + 10
        self.xi_dim = nj + 4
        self._term_dims = (nj, nj, 3, 3, nj)

        model = backend.model
        self._check_model_contract(model)
        self._action_scale = compute_action_scale(model, cfg.action_scale_factor).astype(np.float32)
        joint_range = np.asarray(backend.get_joint_range(), dtype=np.float32)
        mid, half = joint_range.mean(-1), 0.5 * (joint_range[:, 1] - joint_range[:, 0])
        self.soft_joint_limits = np.stack(
            [mid - cfg.soft_joint_limit_factor * half, mid + cfg.soft_joint_limit_factor * half],
            axis=-1,
        )
        self.default_joint_pos = np.asarray(self.default_angles, dtype=np.float32)

        # Bodies: simulator ids of the tracked bodies and their (= MuJoCo body id) npz indices.
        self.body_ids = backend.get_body_ids(cfg.tracked_bodies)
        self.anchor_index = cfg.tracked_bodies.index(cfg.anchor_body_name)
        root_index = self._root_index = cfg.tracked_bodies.index(cfg.root_body_name)
        files = discover_motion_files(cfg.motion_file, cfg.motion_dirs, cfg.motion_manifest, None)
        if cfg.motion_exclude_keywords:
            files = [
                f
                for f in files
                if not any(k in f.rsplit("/", 1)[-1] for k in cfg.motion_exclude_keywords)
            ]
        if cfg.motion_limit:
            files = files[: cfg.motion_limit]
        self.library = SuperMimicMotionLibrary(
            files,
            backend.get_motion_body_ids(cfg.tracked_bodies),
            root_index,
            self.default_joint_pos,
            cfg.bin_frames,
        )
        print(f"[H2SuperMimic] {self.library.summary()}")
        max_steps = (
            int(math.ceil(cfg.max_episode_seconds / cfg.ctrl_dt))
            if cfg.max_episode_seconds
            else 1000
        )
        self.sampler = ReturnBasedSampler(
            self.library, n, cfg.sampling_mode, max_steps, cfg.sampler, self._rng
        )

        # Sim context: foot friction (shared by all foot geoms), encoder bias, torso COM offset.
        pattern = re.compile(cfg.foot_geom_pattern)
        self._foot_geom_ids = np.array(
            [i for i, g in enumerate(backend.get_geom_names()) if g and pattern.match(g)],
            dtype=np.int64,
        )
        if self._foot_geom_ids.size == 0:
            raise ValueError(f"foot_geom_pattern '{cfg.foot_geom_pattern}' matched no geom")
        self._base_geom_friction = backend.get_geom_friction()
        self._base_body_ipos = backend.get_body_ipos()
        self._com_body_id = int(backend.get_body_id(cfg.com_body_name))
        dr = cfg.domain_rand
        self.friction = np.full(n, dr.eval_friction, dtype=np.float32)
        self.encoder_bias = np.zeros((n, nj), dtype=np.float32)
        if dr.enabled:
            lo, hi = np.array(dr.com_offset_range, dtype=np.float32).T
            self.com_offset = self._rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
        else:
            self.com_offset = np.zeros((n, 3), dtype=np.float32)

        # Per-env episode / reference bookkeeping.
        self.frame = np.zeros(n, dtype=np.int64)  # current target frame (global)
        self.clip_end = np.zeros(n, dtype=bool)
        self._clip_end_at_step = np.zeros(n, dtype=bool)
        self.forced_frame = np.full(
            n, -1, dtype=np.int64
        )  # evaluation hook: next reset of env i starts here
        self.action = np.zeros((n, nj), dtype=np.float32)
        self.prev_action = np.zeros((n, nj), dtype=np.float32)
        self.sm_executed_action = np.zeros((n, nj), dtype=np.float32)
        self._obs_hist = np.zeros((cfg.history_length, n, self.obs_frame_dim), dtype=np.float32)
        noise = [
            np.full(d, h, dtype=np.float32) for d, h in zip(self._term_dims, cfg.obs_noise_half)
        ]
        self._obs_noise_half = (
            np.concatenate(noise)
            if cfg.obs_noise
            else np.zeros(self.obs_frame_dim, dtype=np.float32)
        )
        # Snapshots of the step that just finished, taken before auto-reset (for the replay buffer).
        self.sm_state_pre_reset = np.zeros((n, self.state_dim), dtype=np.float32)
        self.sm_frame_pre_reset = np.zeros(n, dtype=np.int64)
        self.sm_reward_terms: dict[str, np.ndarray] = {}

        self._init_domain_randomization(SuperMimicResetProvider())

    # ------------------------------------------------------------------ model contract

    def _check_model_contract(self, model: Any) -> None:
        import mujoco

        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        if (
            model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or tuple(names[1:]) != self._cfg.joint_names
        ):
            raise ValueError(f"H2SuperMimic: joint order mismatch: {names}")
        acts = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
            for i in range(model.nu)
        ]
        if tuple(acts) != self._cfg.joint_names:
            raise ValueError(f"H2SuperMimic: actuator order mismatch: {acts}")

    @property
    def action_scale(self) -> np.ndarray:
        return self._action_scale

    # ------------------------------------------------------------------ spaces

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": self._cfg.history_length * self.obs_frame_dim,
            "goal": self._cfg.future_frames * self.goal_frame_dim,
        }

    # ------------------------------------------------------------------ state readout

    def sm_state(self, rows: np.ndarray | None = None) -> np.ndarray:
        """(n, 13+2N) simulator state ``[root_pos, root_quat, q, root_lin_vel_w, root_ang_vel_b, qd]``.

        MuJoCo free-joint conventions (``qpos | qvel``); the MuJoCo backend exposes ``qvel[3:6]``,
        i.e. the body-frame angular velocity, through ``get_base_ang_vel``.
        """
        b = self._backend
        s = np.concatenate(
            [
                b.get_base_pos(),
                b.get_base_quat(),
                b.get_dof_pos(),
                b.get_base_lin_vel(),
                b.get_base_ang_vel(),
                b.get_dof_vel(),
            ],
            axis=-1,
        ).astype(np.float32)
        return s if rows is None else s[rows]

    def sm_xi(self, rows: np.ndarray | None = None) -> np.ndarray:
        """(n, N+4) sim context ``[foot friction, encoder bias (N), COM offset (3)]``."""
        xi = np.concatenate([self.friction[:, None], self.encoder_bias, self.com_offset], axis=-1)
        return xi if rows is None else xi[rows]

    # ------------------------------------------------------------------ action

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        a = np.asarray(actions, dtype=np.float32)
        self.prev_action[:] = self.action
        self.action[:] = a
        if self._cfg.action_noise_std > 0:
            self.sm_executed_action = a + self._cfg.action_noise_std * self._rng.standard_normal(
                a.shape
            ).astype(np.float32)
        else:
            self.sm_executed_action = a.copy()
        state.info["current_actions"] = a
        return (
            self.default_joint_pos
            + self.sm_executed_action * self._action_scale
            - self.encoder_bias
        )

    # ------------------------------------------------------------------ observation

    def _proprio(self, s: np.ndarray, last_action: np.ndarray, bias: np.ndarray) -> np.ndarray:
        nj = self.num_joints
        q = s[:, 7 : 7 + nj]
        quat = s[:, 3:7]
        ang_b = s[:, 10 + nj : 13 + nj]
        qd = s[:, 13 + nj :]
        grav = quat_apply_inverse(quat, np.broadcast_to(_GRAVITY_W, (s.shape[0], 3)))
        o = np.concatenate(
            [q + bias - self.default_joint_pos, qd, ang_b, grav, last_action], axis=-1
        )
        if self._cfg.obs_noise:
            o = o + self._rng.uniform(-1.0, 1.0, o.shape).astype(np.float32) * self._obs_noise_half
        return o.astype(np.float32)

    def _flatten_history(self, rows: np.ndarray | slice) -> np.ndarray:
        hist = self._obs_hist[:, rows]  # (H, n, D) oldest -> newest
        H, m, _ = hist.shape
        chunks, start = [], 0
        for d in self._term_dims:
            chunks.append(hist[:, :, start : start + d].transpose(1, 0, 2).reshape(m, H * d))
            start += d
        return np.concatenate(chunks, axis=-1)

    def _observation(self, rows: np.ndarray | slice) -> dict[str, np.ndarray]:
        return {
            "obs": self._flatten_history(rows),
            "goal": self.library.goal_window(self.frame[rows], self._cfg.future_frames),
        }

    # ------------------------------------------------------------------ step

    def update_state(self, state: NpEnvState) -> NpEnvState:
        cfg = self._cfg
        s = self.sm_state()
        ref = self.library.ref(self.frame)
        pos, quat, lin, ang = self._backend.get_body_state_w(self.body_ids)

        # Terminations: pelvis height and gravity tilt versus the reference. Like mjlab, termination
        # and reward read derived body poses, which lag qpos by one physics substep (mj_step runs
        # forward kinematics before integrating); observations and ``s`` use the current qpos/qvel.
        nj = self.num_joints
        root_pos, root_quat = pos[:, self._root_index], quat[:, self._root_index]
        g = np.broadcast_to(_GRAVITY_W, root_pos.shape)
        tilt = quat_apply_inverse(ref.root_quat_w, g)[:, 2] - quat_apply_inverse(root_quat, g)[:, 2]
        terminated = (np.abs(ref.root_pos_w[:, 2] - root_pos[:, 2]) > cfg.root_height_threshold) | (
            np.abs(tilt) > cfg.root_tilt_threshold
        )

        terms = tracking_reward_terms(
            robot_body_pos_w=pos,
            robot_body_quat_w=quat,
            robot_body_lin_vel_w=lin,
            robot_body_ang_vel_w=ang,
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=self.action,
            prev_action=self.prev_action,
            joint_pos=s[:, 7 : 7 + nj],
            soft_limits=self.soft_joint_limits,
            cfg=cfg.reward,
            dt=cfg.ctrl_dt,
        )
        reward = terms.pop("total")
        self.sm_reward_terms = terms
        if self.step_counter % 4 == 0:
            state.info["log"] = {f"reward/{k}": float(v.mean()) for k, v in terms.items()}
        self.sampler.log_step(reward, self.frame)

        # Snapshot of s_{t+1} and its target frame before any auto-reset overwrites them.
        self.sm_state_pre_reset[:] = s
        self.sm_frame_pre_reset[:] = self.frame
        self._clip_end_at_step[:] = self.clip_end

        # Advance the target frame (envs that reset below get their own frame).
        self._advance_frames(np.arange(self._num_envs))
        self._obs_hist[:-1] = self._obs_hist[1:]
        self._obs_hist[-1] = self._proprio(s, self.action, self.encoder_bias)
        obs = self._observation(slice(None))
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _advance_frames(self, rows: np.ndarray) -> None:
        last = self.library.last_frame_of(self.frame[rows])
        nxt = self.frame[rows] + 1
        over = nxt > last
        self.clip_end[rows] |= over
        self.frame[rows] = np.where(over, last, nxt)

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        np.logical_or(truncated, self._clip_end_at_step & ~state.terminated, out=truncated)
        return truncated

    # ------------------------------------------------------------------ reset

    def _build_reset_plan(self, env_ids: np.ndarray) -> ResetPlan:
        cfg = self._cfg
        m = len(env_ids)
        if self._state is not None:
            self.sampler.finalize(
                env_ids, self._state.terminated[env_ids].copy(), self.clip_end[env_ids].copy()
            )
        f = self.sampler.sample(m)
        forced = self.forced_frame[env_ids]
        f = np.where(forced >= 0, forced, f)
        self.frame[env_ids] = f
        self.clip_end[env_ids] = False
        r = self.library.ref(f)

        root_pos = r.root_pos_w.copy()
        root_quat = r.root_quat_w.copy()
        lin = r.root_lin_vel_w.copy()
        ang = r.root_ang_vel_w.copy()
        q = r.joint_pos.copy()
        if cfg.reset_noise.enabled:
            lo, hi = np.array(cfg.reset_noise.pose, dtype=np.float32).T
            p = self._rng.uniform(lo, hi, size=(m, 6)).astype(np.float32)
            root_pos += p[:, :3]
            root_quat = np_quat_mul(
                np_quat_from_euler_xyz(p[:, 3], p[:, 4], p[:, 5]), root_quat
            ).astype(np.float32)
            lo, hi = np.array(cfg.reset_noise.velocity, dtype=np.float32).T
            v = self._rng.uniform(lo, hi, size=(m, 6)).astype(np.float32)
            lin += v[:, :3]
            ang += v[:, 3:]
            jlo, jhi = cfg.reset_noise.joint_pos
            q += self._rng.uniform(jlo, jhi, size=q.shape).astype(np.float32)
        q = np.clip(q, self.soft_joint_limits[:, 0], self.soft_joint_limits[:, 1])
        qpos = np.concatenate([root_pos, root_quat, q], axis=-1)
        qvel = np.concatenate([lin, quat_apply_inverse(root_quat, ang), r.joint_vel], axis=-1)

        # Domain randomisation (per episode) and the per-env COM offset (re-applied every reset).
        dr = cfg.domain_rand
        if dr.enabled:
            self.friction[env_ids] = self._rng.uniform(*dr.friction_range, size=m).astype(
                np.float32
            )
            self.encoder_bias[env_ids] = self._rng.uniform(
                *dr.encoder_bias_range, size=(m, self.num_joints)
            ).astype(np.float32)
        geom_friction = np.broadcast_to(
            self._base_geom_friction, (m, *self._base_geom_friction.shape)
        ).copy()
        geom_friction[:, self._foot_geom_ids, 0] = self.friction[env_ids][:, None]
        body_ipos = np.broadcast_to(self._base_body_ipos, (m, *self._base_body_ipos.shape)).copy()
        body_ipos[:, self._com_body_id] += self.com_offset[env_ids]
        payload = ResetRandomizationPayload(geom_friction=geom_friction, body_ipos=body_ipos)

        self.action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        zeros = np.zeros((m, self.num_joints), dtype=np.float32)
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates={"current_actions": zeros, "last_actions": zeros},
            randomization=payload,
        )

    def _reset_observation(self, env_ids: np.ndarray) -> dict[str, np.ndarray]:
        # mjlab timing: the target moves to f + 1 right after the reset to frame f.
        self._advance_frames(env_ids)
        o = self._proprio(self.sm_state(env_ids), self.action[env_ids], self.encoder_bias[env_ids])
        self._obs_hist[:, env_ids] = o[None]  # history buffer restarts filled with the first frame
        return self._observation(env_ids)


__all__ = [
    "H2_JOINT_NAMES",
    "H2_TRACKED_BODIES",
    "H2SuperMimicCfg",
    "H2SuperMimicEnv",
    "OBS_TERMS",
    "SuperMimicRewardCfg",
    "collision_only_scene",
    "compute_action_scale",
    "reanchor_reference",
    "tracking_reward_terms",
]
