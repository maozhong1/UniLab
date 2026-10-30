from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from scripts.motion.g1_csv_to_gr2_npz import _scalar_joints, _validate_target
from tensordict import TensorDict

from unilab.algos.torch.sonic import SonicGR2ActorModel
from unilab.envs.motion_tracking.gr2 import (
    GR2SonicMotionTrackingCfg,
    GR2SonicMotionTrackingEnv,
)

_ASSET_DIR = Path(__file__).parents[1] / "src" / "unilab" / "assets" / "robots" / "gr2"


def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [name for name, *_ in _scalar_joints(model)]


def _actuator_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[index, 0]))
        for index in range(model.nu)
    ]


def test_gr2_sonic_profiles_preserve_order_and_fixed_head() -> None:
    full = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_flat.xml"))
    sonic = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_sonic_27dof.xml"))

    assert (full.nq, full.nv, full.nu) == (36, 35, 29)
    assert (sonic.nq, sonic.nv, sonic.nu) == (34, 33, 27)
    assert _joint_names(full) == _actuator_joint_names(full)
    assert _joint_names(sonic) == _actuator_joint_names(sonic)
    assert {"head_yaw_link", "head_pitch_link"} <= {
        mujoco.mj_id2name(sonic, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(sonic.nbody)
    }
    assert not {"head_yaw_joint", "head_pitch_joint"}.intersection(_joint_names(sonic))
    assert len(_validate_target(sonic)) == 27
    assert len(_validate_target(full)) == 29


def test_gr2_sonic_stand_keyframe_is_finite_and_in_range() -> None:
    model = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_sonic_27dof.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.ctrl).all()
    assert np.all(data.qpos[7:] >= model.jnt_range[:, 0][1:])
    assert np.all(data.qpos[7:] <= model.jnt_range[:, 1][1:])
    assert np.isfinite(data.xpos).all()


def test_gr2_sonic_uses_reference_sole_collision_baseline() -> None:
    model = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_sonic_27dof.xml"))

    for side in ("l", "r"):
        sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_sole")
        np.testing.assert_allclose(model.geom_pos[sole_id], [0.06, 0.0, -0.048])
        np.testing.assert_allclose(model.geom_size[sole_id], [0.14, 0.05, 0.01])
        np.testing.assert_allclose(model.geom_friction[sole_id], [1.0, 0.005, 0.0001])
        assert model.geom_condim[sole_id] == 3

    for body_name in (
        "left_foot_pitch_link",
        "left_foot_roll_link",
        "right_foot_pitch_link",
        "right_foot_roll_link",
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        mesh_geom_ids = body_geom_ids[model.geom_type[body_geom_ids] == mujoco.mjtGeom.mjGEOM_MESH]
        assert np.all(model.geom_contype[mesh_geom_ids] == 0)
        assert np.all(model.geom_conaffinity[mesh_geom_ids] == 0)


def test_gr2_sonic_actor_forward_shape() -> None:
    obs = TensorDict({"actor": torch.zeros(2, 1470)}, batch_size=[2])
    actor = SonicGR2ActorModel(
        obs,
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=27,
        enc_group="actor",
        proprio_group=None,
        use_fsq=False,
    )

    assert actor(obs).shape == (2, 27)


def test_gr2_sonic_action_scale_matches_actuator_contract() -> None:
    cfg = GR2SonicMotionTrackingCfg()
    model = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_sonic_27dof.xml"))
    effort = model.actuator_forcerange[:, 1]
    kp = model.actuator_gainprm[:, 0]
    kd = -model.actuator_biasprm[:, 2]

    arm_kp = [300, 300, 100, 100, 50, 50, 50]
    arm_kd = [10, 10, 5, 5, 5, 5, 5]
    leg_kp = [90, 180, 120, 90, 30, 60]
    leg_kd = [19, 10, 9, 19, 5, 3.5]
    np.testing.assert_allclose(kp, [200, *arm_kp, *arm_kp, *leg_kp, *leg_kp])
    np.testing.assert_allclose(kd, [10, *arm_kd, *arm_kd, *leg_kd, *leg_kd])

    np.testing.assert_allclose(cfg.control_config.action_scale, 0.25 * effort / kp)


def test_gr2_sonic_target_rate_uses_physical_position_targets() -> None:
    env = GR2SonicMotionTrackingEnv(GR2SonicMotionTrackingCfg(), num_envs=1)
    current = np.linspace(-1.0, 1.0, 27, dtype=np.float32)[None, :]
    last = np.linspace(0.5, -0.5, 27, dtype=np.float32)[None, :]
    ctx = SimpleNamespace(
        info={"current_actions": current, "last_actions": last},
        joint_error=np.empty_like(current),
        env_error=np.empty((1,), dtype=np.float32),
    )

    actual = env._reward_target_rate_l2(ctx).copy()
    expected = np.sum(np.square((current - last) * env._cfg.control_config.action_scale), axis=1)

    np.testing.assert_allclose(actual, expected)


def test_gr2_sonic_upper_body_reward_covers_14_arm_joints() -> None:
    env = GR2SonicMotionTrackingEnv(GR2SonicMotionTrackingCfg(), num_envs=1)
    reference = np.zeros((1, 27), dtype=np.float32)
    actual_pos = np.zeros_like(reference)
    actual_pos[:, env._upper_body_dof_indices] = 0.2
    ctx = SimpleNamespace(
        motion_data=SimpleNamespace(joint_pos=reference),
        dof_pos=actual_pos,
        env_error=np.empty((1,), dtype=np.float32),
        reward_term=np.empty((1,), dtype=np.float32),
    )

    actual = env._reward_upper_body_joint_pos(ctx).copy()

    assert env._upper_body_dof_indices.size == 14
    np.testing.assert_allclose(actual, [np.exp(-1.0)], rtol=1e-6)


def test_gr2_sonic_linear_velocity_sensor_uses_imu_frame() -> None:
    model = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_sonic_27dof.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    data.qvel[:3] = [1.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "baselink-velocity"
    )
    sensor_adr = model.sensor_adr[sensor_id]
    np.testing.assert_allclose(
        data.sensordata[sensor_adr : sensor_adr + 3],
        [0.0, -1.0, 0.0],
        atol=1e-7,
    )


def test_gr2_sonic_env_reset_and_step_shapes_are_finite() -> None:
    cfg = GR2SonicMotionTrackingCfg(critic_privileged_mf_hist=True)
    env = GR2SonicMotionTrackingEnv(cfg, num_envs=1, backend_type="mujoco")

    obs, _ = env.reset(np.array([0], dtype=np.int32))
    assert env.action_space.shape == (27,)
    assert np.isfinite(env.action_space.low).all()
    assert np.isfinite(env.action_space.high).all()
    assert env.obs_groups_spec == {"obs": 1470, "critic": 1545}
    assert obs["obs"].shape == (1, 1470)
    assert obs["critic"].shape == (1, 1545)
    assert all(np.isfinite(value).all() for value in obs.values())

    state = env.step(np.zeros((1, 27), dtype=np.float32))
    assert state.obs["obs"].shape == (1, 1470)
    assert state.obs["critic"].shape == (1, 1545)
    assert all(np.isfinite(value).all() for value in state.obs.values())
    assert np.isfinite(state.reward).all()
    assert {
        "termination/anchor_height",
        "termination/anchor_tilt",
        "termination/end_effector_height",
        "termination/undesired_contact",
        "termination/strict_anchor_height",
        "termination/strict_end_effector_height",
        "termination/strict_anchor_orientation",
        "termination/strict_foot_position",
        "termination/any",
        "truncation/time_limit",
        "truncation/clip_end",
        "truncation/any",
        "episode_end/any",
        "diagnostic/estimated_torque_saturation_ratio",
        "diagnostic/estimated_upper_body_torque_saturation_ratio",
    } <= state.info["log"].keys()


def _sole_floor_contact_friction(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    sole_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_sole") for side in "lr"
    }
    return [
        float(contact.friction[0])
        for contact in data.contact[: data.ncon]
        if {int(contact.geom1), int(contact.geom2)} & sole_ids
        and floor_id in (contact.geom1, contact.geom2)
    ]


def test_gr2_sonic_sole_friction_owns_floor_contact() -> None:
    model = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_sonic_27dof.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 0.92
    for _ in range(200):
        mujoco.mj_step(model, data)
        if _sole_floor_contact_friction(model, data):
            break

    nominal = _sole_floor_contact_friction(model, data)
    assert nominal
    np.testing.assert_allclose(nominal, 1.0)

    for side in "lr":
        sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_sole")
        model.geom_friction[sole_id, 0] = 0.3
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(_sole_floor_contact_friction(model, data), 0.3)


def _reset_obs(cfg: GR2SonicMotionTrackingCfg) -> dict[str, np.ndarray]:
    env = GR2SonicMotionTrackingEnv(cfg, num_envs=1, backend_type="mujoco")
    np.random.seed(0)
    obs, _ = env.reset(np.array([0], dtype=np.int32))
    return obs


def test_gr2_sonic_actor_noise_is_bounded_and_critic_stays_clean() -> None:
    clean = _reset_obs(GR2SonicMotionTrackingCfg(critic_privileged_mf_hist=True))
    noisy_cfg = GR2SonicMotionTrackingCfg(critic_privileged_mf_hist=True)
    noisy_cfg.noise_config.level = 0.5
    noisy_cfg.noise_config.seed = 0
    noisy = _reset_obs(noisy_cfg)

    np.testing.assert_array_equal(noisy["critic"], clean["critic"])
    np.testing.assert_array_equal(noisy["obs"][:, :600], clean["obs"][:, :600])

    n, history = 27, noisy_cfg.proprio_history_len
    noise = noisy_cfg.noise_config
    diff = (noisy["obs"] - clean["obs"])[0, 600:]
    offset = 0
    for width, scale in (
        (3, noise.scale_gyro),
        (n, noise.scale_joint_angle),
        (n, noise.scale_joint_vel),
        (n, 0.0),
        (3, noise.scale_gravity),
    ):
        term = diff[offset : offset + width * history]
        offset += width * history
        assert np.all(np.abs(term) <= 0.5 * scale + 1e-6)
        if scale > 0.0:
            assert np.any(term != 0.0)
    assert offset == diff.size


def test_gr2_sonic_foot_friction_randomization_matches_soles() -> None:
    cfg = GR2SonicMotionTrackingCfg(critic_privileged_mf_hist=True)
    cfg.domain_rand.randomize_geom_friction = True
    cfg.domain_rand.friction_range = [0.3, 0.3]
    env = GR2SonicMotionTrackingEnv(cfg, num_envs=1, backend_type="mujoco")
    assert env._dr_manager is not None
    plan = env._dr_manager._provider.build_reset_plan(env, np.array([0], dtype=np.int32))

    assert plan.randomization is not None
    friction = plan.randomization.geom_friction
    base = env._backend.get_geom_friction()
    names = env._backend.get_geom_names()
    sole_ids = [names.index(f"{side}_foot_sole") for side in "lr"]
    np.testing.assert_allclose(friction[0, sole_ids, 0], 0.3)
    other = np.ones(len(names), dtype=bool)
    other[sole_ids] = False
    np.testing.assert_array_equal(friction[0, other], base[other])


def test_gr2_sonic_joint_armature_randomization_skips_floating_base() -> None:
    cfg = GR2SonicMotionTrackingCfg(critic_privileged_mf_hist=True)
    cfg.domain_rand.randomize_joint_armature = True
    cfg.domain_rand.joint_armature_multiplier_range = [0.8, 1.2]
    env = GR2SonicMotionTrackingEnv(cfg, num_envs=1, backend_type="mujoco")
    assert env._dr_manager is not None
    np.random.seed(0)
    plan = env._dr_manager._provider.build_reset_plan(env, np.arange(1, dtype=np.int32))

    assert plan.randomization is not None
    armature = plan.randomization.dof_armature[0]
    base = env._backend.get_dof_armature()
    assert base.size == 6 + 27
    np.testing.assert_array_equal(armature[:6], base[:6])
    ratio = armature[6:] / base[6:]
    assert np.all((ratio >= 0.8) & (ratio <= 1.2))
    assert np.unique(ratio).size > 1

    env.reset(np.arange(1, dtype=np.int32))
    state = env.step(np.zeros((1, 27), dtype=np.float32))
    assert all(np.isfinite(value).all() for value in state.obs.values())


def test_gr2_sonic_physics_scene_matches_full_scene_dynamics_contract() -> None:
    full = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_sonic_27dof.xml"))
    light = mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "scene_sonic_27dof_physics.xml"))

    assert (light.nq, light.nv, light.nu, light.nbody, light.nsensor) == (
        full.nq, full.nv, full.nu, full.nbody, full.nsensor,
    )
    for field in (
        "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
        "jnt_range", "dof_armature", "dof_damping", "dof_frictionloss",
        "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "sensor_type",
        "key_qpos", "key_ctrl",
    ):
        np.testing.assert_allclose(getattr(light, field), getattr(full, field), err_msg=field)
    assert (light.opt.cone, light.opt.solver) == (full.opt.cone, full.opt.solver)

    def collision_geoms(model: mujoco.MjModel) -> list[tuple]:
        return [
            (
                int(model.geom_bodyid[i]), int(model.geom_type[i]),
                int(model.geom_contype[i]), int(model.geom_conaffinity[i]),
                int(model.geom_condim[i]), int(model.geom_priority[i]),
                *np.round(model.geom_friction[i], 6), *np.round(model.geom_solref[i], 6),
            )
            for i in range(model.ngeom)
            if model.geom_contype[i] or model.geom_conaffinity[i]
        ]

    assert collision_geoms(light) == collision_geoms(full)
    for name in ("floor", "l_foot_sole", "r_foot_sole"):
        full_id = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_GEOM, name)
        light_id = mujoco.mj_name2id(light, mujoco.mjtObj.mjOBJ_GEOM, name)
        np.testing.assert_allclose(light.geom_size[light_id], full.geom_size[full_id])
        np.testing.assert_allclose(light.geom_pos[light_id], full.geom_pos[full_id])
    assert light.ntex == 0
    assert light.nmeshface < full.nmeshface // 20


def test_gr2_sonic_action_delay_holds_previous_target_for_leading_substeps() -> None:
    cfg = GR2SonicMotionTrackingCfg(critic_privileged_mf_hist=True)
    cfg.domain_rand.randomize_action_delay = True
    cfg.domain_rand.action_delay_substeps = [2, 2]
    cfg.sim_dt = 0.005
    env = GR2SonicMotionTrackingEnv(cfg, num_envs=2, backend_type="mujoco")
    env.init_state()
    np.testing.assert_array_equal(env._state.info["action_delay_substeps"], [2, 2])

    previous = np.full((2, 27), 0.3, dtype=np.float32)
    current = np.full((2, 27), -0.4, dtype=np.float32)
    env._state.info["current_actions"] = previous
    trajectory = env.apply_action(current, env._state)

    scale = env._cfg.control_config.action_scale
    assert cfg.sim_substeps == 4
    assert trajectory.shape == (2, 4, 27)
    for substep in range(2):
        np.testing.assert_allclose(
            trajectory[:, substep], previous * scale + env.default_angles, rtol=1e-6
        )
    for substep in range(2, 4):
        np.testing.assert_allclose(
            trajectory[:, substep], current * scale + env.default_angles, rtol=1e-6
        )


def test_mujoco_constant_control_trajectory_matches_held_control() -> None:
    from unilab.base.backend import create_backend
    from unilab.base.scene import SceneCfg

    scene = SceneCfg(model_file=str(_ASSET_DIR / "scene_sonic_27dof.xml"))
    results = []
    for as_trajectory in (False, True):
        backend = create_backend("mujoco", scene, 2, 0.005)
        backend.materialize()
        ctrl = np.tile(np.linspace(-0.2, 0.2, 27), (2, 1))
        if as_trajectory:
            ctrl = np.repeat(ctrl[:, None, :], 4, axis=1)
        backend.step(ctrl, 4)
        results.append(backend.get_dof_pos().copy())
    assert backend.supports_control_trajectory
    np.testing.assert_array_equal(results[0], results[1])
    with pytest.raises(ValueError, match="control trajectory"):
        backend.step(np.zeros((2, 3, 27)), 4)


def test_gr2_sonic_random_action_delay_steps_and_validates_range() -> None:
    cfg = GR2SonicMotionTrackingCfg(critic_privileged_mf_hist=True)
    cfg.domain_rand.randomize_action_delay = True
    cfg.domain_rand.action_delay_substeps = [1, 4]
    cfg.sim_dt = 0.005
    env = GR2SonicMotionTrackingEnv(cfg, num_envs=64, backend_type="mujoco")
    np.random.seed(0)
    env.init_state()
    delay = env._state.info["action_delay_substeps"]
    assert delay.min() >= 1 and delay.max() <= 4 and np.unique(delay).size > 1
    for _ in range(3):
        state = env.step(np.zeros((64, 27), dtype=np.float32))
    assert all(np.isfinite(value).all() for value in state.obs.values())

    bad = GR2SonicMotionTrackingCfg()
    bad.domain_rand.randomize_action_delay = True
    bad.domain_rand.action_delay_substeps = [1, 5]
    bad.sim_dt = 0.005
    with pytest.raises(ValueError, match="action_delay_substeps"):
        GR2SonicMotionTrackingEnv(bad, num_envs=1, backend_type="mujoco")
