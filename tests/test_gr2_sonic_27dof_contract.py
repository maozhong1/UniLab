from pathlib import Path

import mujoco
import numpy as np
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
