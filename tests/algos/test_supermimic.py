"""SuperMimic port: torch (imagination) side versus MuJoCo / the H2SuperMimic env (simulator side)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from unilab.algos.torch.supermimic import SuperMimicCfg, build_models
from unilab.algos.torch.supermimic.heads import analytic_observation, flatten_history_term_major
from unilab.algos.torch.supermimic.imagination import gae_targets
from unilab.algos.torch.supermimic.kinematics import HumanoidKinematics
from unilab.algos.torch.supermimic.replay_buffer import ReplayBuffer
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.envs.motion_tracking.h2.supermimic import (
    H2SuperMimicCfg,
    compute_action_scale,
    tracking_reward_terms,
)

SCENE = str(ASSETS_ROOT_PATH / "robots" / "h2" / "scene_flat_supermimic.xml")


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(SCENE)


def _random_state(model: mujoco.MjModel, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    qpos = model.key_qpos[0].copy()
    qpos[7:] += rng.uniform(-0.3, 0.3, model.nq - 7)
    quat = rng.normal(size=4)
    qpos[3:7] = quat / np.linalg.norm(quat)
    qvel = rng.normal(0.0, 1.0, model.nv)
    return qpos, qvel


def _write_synthetic_clip(
    model: mujoco.MjModel, path, num_frames: int = 120, phase: float = 0.0
) -> None:
    """A smooth 50 fps clip around the stand pose, exported like ``g1_csv_to_h2_npz2.py`` (all bodies, MuJoCo order)."""
    data = mujoco.MjData(model)
    t = np.arange(num_frames) / 50.0
    qpos = np.repeat(model.key_qpos[0][None], num_frames, axis=0)
    qpos[:, 0] += 0.3 * t
    qpos[:, 7:] += 0.15 * np.sin(2.0 * t[:, None] + phase + np.arange(model.nq - 7)[None])
    qvel = np.zeros((num_frames, model.nv))
    for i in range(num_frames):
        a, b = qpos[max(i - 1, 0)], qpos[min(i + 1, num_frames - 1)]
        mujoco.mj_differentiatePos(
            model, qvel[i], (min(i + 1, num_frames - 1) - max(i - 1, 0)) / 50.0, a, b
        )
    out = {
        k: np.zeros((num_frames, model.nbody, d), np.float32)
        for k, d in (("pos", 3), ("quat", 4), ("lin", 3), ("ang", 3))
    }
    v6 = np.zeros(6)
    for i in range(num_frames):
        data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(model, data)
        out["pos"][i], out["quat"][i] = data.xpos, data.xquat
        for b in range(model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, v6, 0)
            out["ang"][i, b], out["lin"][i, b] = v6[:3], v6[3:]
    np.savez(
        path,
        fps=np.array([50], dtype=np.int32),
        joint_pos=qpos[:, 7:].astype(np.float32),
        joint_vel=qvel[:, 6:].astype(np.float32),
        body_pos_w=out["pos"],
        body_quat_w=out["quat"],
        body_lin_vel_w=out["lin"],
        body_ang_vel_w=out["ang"],
    )


@pytest.fixture(scope="module")
def clips(model, tmp_path_factory) -> list[str]:
    d = tmp_path_factory.mktemp("h2_clips")
    files = []
    for i, n in enumerate((120, 80)):
        p = d / f"clip{i}.npz"
        _write_synthetic_clip(model, p, num_frames=n, phase=float(i))
        files.append(str(p))
    return files


def _make_env(clips, num_envs=4, **over):
    registry.ensure_registries()
    cfg = {"motion_file": clips, "seed": 0, **over}
    return registry.make(
        "H2SuperMimic", sim_backend="mujoco", num_envs=num_envs, env_cfg_override=cfg
    )


def _small_cfg() -> SuperMimicCfg:
    cfg = SuperMimicCfg()
    cfg.world_model.hidden = (32,)
    cfg.policy.encoder_hidden = (32,)
    cfg.policy.decoder_hidden = (32,)
    cfg.critic.hidden = (16,)
    return cfg


# ---------------------------------------------------------------------- pure torch side


def test_fk_matches_mujoco(model):
    rng = np.random.default_rng(0)
    fk = HumanoidKinematics(model, dtype=torch.float64)
    data = mujoco.MjData(model)
    v6 = np.zeros(6)
    for _ in range(3):
        qpos, qvel = _random_state(model, rng)
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
        out = fk(torch.tensor(qpos)[None], torch.tensor(qvel)[None])
        np.testing.assert_allclose(out["pos"][0].numpy(), data.xpos[1:], atol=1e-9)
        dots = np.abs((out["quat"][0].numpy() * data.xquat[1:]).sum(-1))
        np.testing.assert_allclose(dots, 1.0, atol=1e-9)
        for b in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, v6, 0)
            np.testing.assert_allclose(out["ang_vel"][0, b - 1].numpy(), v6[:3], atol=1e-9)
            np.testing.assert_allclose(out["lin_vel"][0, b - 1].numpy(), v6[3:], atol=1e-9)


def test_action_scale_matches_reference(model):
    scale = compute_action_scale(model, 0.25)
    # SuperMimic H2_TRAINING_README: ankle roll 0.119, ankle pitch 0.239, hip/knee 0.45, waist 0.2,
    # arms 0.338, wrist pitch / yaw 0.188.
    np.testing.assert_allclose(
        scale[[0, 3, 4, 5, 12, 15, 20, 21]],
        [0.45, 0.45, 0.119, 0.239, 0.2, 0.338, 0.188, 0.188],
        atol=1e-3,
    )


def test_flatten_history_is_term_major():
    nj, H, n = 2, 3, 1
    hist = torch.arange(H * n * (3 * nj + 6), dtype=torch.float32).reshape(H, n, -1)
    flat = flatten_history_term_major(hist)[0]
    np.testing.assert_array_equal(flat[: H * nj].numpy(), hist[:, 0, :nj].reshape(-1).numpy())
    np.testing.assert_array_equal(flat[-H * nj :].numpy(), hist[:, 0, -nj:].reshape(-1).numpy())


def test_gae_targets_reduce_to_discounted_return():
    r = torch.ones(3, 1)
    v = torch.zeros(4, 1)
    out = gae_targets(r, torch.zeros(3, 1), v, gamma=0.5, lam=1.0)
    np.testing.assert_allclose(out[:, 0].numpy(), [1.75, 1.5, 1.0])


def test_replay_windows_stay_inside_episodes():
    buf = ReplayBuffer(num_envs=1, capacity=32, state_dim=1, act_dim=1, xi_dim=1, device="cpu")
    ids = torch.tensor([0])
    for t in range(20):
        terminal = t == 9
        buf.add(
            ids,
            s=torch.full((1, 1), float(t)),
            a=torch.zeros(1, 1),
            a_exec=torch.zeros(1, 1),
            xi=torch.zeros(1, 1),
            r=torch.zeros(1),
            done=torch.zeros(1),
            frame=torch.zeros(1, dtype=torch.long),
            clip=torch.zeros(1, dtype=torch.long),
            terminal=torch.tensor([terminal]),
            ep_start=torch.tensor([t in (0, 10)]),
        )
        if terminal:
            buf.end_episode(ids)
    w = buf.sample_windows(256, 4)
    first = w.states[0, :, 0]
    assert bool(
        ((first <= 5) | (first >= 10)).all()
    )  # windows of episode 0 end at the terminal row 9


# ---------------------------------------------------------------------- env <-> torch parity


def test_env_contract_and_reset_timing(clips):
    env = _make_env(clips, sampling_mode="start", reset_noise={"enabled": False})
    st = env.init_state()
    assert st.obs["obs"].shape == (4, 930) and st.obs["goal"].shape == (4, 680)
    assert env.sm_state().shape == (4, 71) and env.sm_xi().shape == (4, 33)
    lib = env.library
    # Reset at a clip start f -> the first target is f + 1 (mjlab timing).
    assert set((env.frame - lib.clip_start[lib.frame_clip[env.frame]]).tolist()) == {1}
    s = env.sm_state()
    ref = lib.ref(env.frame - 1)
    np.testing.assert_allclose(
        s[:, 7:36],
        np.clip(ref.joint_pos, env.soft_joint_limits[:, 0], env.soft_joint_limits[:, 1]),
        atol=1e-5,
    )
    frame0 = env.frame.copy()
    st = env.step(np.zeros((4, 29), np.float32))
    alive = ~(st.terminated | st.truncated)
    np.testing.assert_array_equal(env.frame[alive], frame0[alive] + 1)


def test_env_matches_torch_heads(clips):
    env = _make_env(clips, obs_noise=False, action_noise_std=0.0)
    wm, _, _, lib = build_models(env, _small_cfg(), "cpu")
    env.init_state()
    rng = np.random.default_rng(1)
    for _ in range(3):
        st = env.step(rng.normal(0.0, 0.2, (4, 29)).astype(np.float32))
    s = torch.tensor(env.sm_state())
    xi = torch.tensor(env.sm_xi())
    # Observation: newest history frame and the flattened history.
    obs = analytic_observation(s, torch.tensor(env.action), xi, lib.default_joint_pos)
    np.testing.assert_allclose(env._obs_hist[-1], obs.numpy(), atol=1e-5)
    np.testing.assert_allclose(
        flatten_history_term_major(torch.tensor(env._obs_hist)).numpy(), st.obs["obs"], atol=1e-6
    )
    # Goal features and reference bodies (npz versus FK of the reference state).
    frame = torch.tensor(env.frame)
    np.testing.assert_allclose(lib.goal_window(frame, 10).numpy(), st.obs["goal"], atol=1e-5)
    ref = lib.reference_at(frame)
    env_ref = env.library.ref(env.frame)
    np.testing.assert_allclose(ref.body_pos_w.numpy(), env_ref.body_pos_w, atol=1e-5)
    np.testing.assert_allclose(ref.body_lin_vel_w.numpy(), env_ref.body_lin_vel_w, atol=1e-4)
    # Reward: torch heads versus the env's NumPy reward on identical inputs.
    b = wm.heads.robot_bodies(s)
    a, pa = torch.tensor(env.action), torch.tensor(env.prev_action)
    with torch.no_grad():
        t_terms = wm.heads.reward(s, a, pa, ref, b)
    n_terms = tracking_reward_terms(
        robot_body_pos_w=b["pos"].numpy(),
        robot_body_quat_w=b["quat"].numpy(),
        robot_body_lin_vel_w=b["lin_vel"].numpy(),
        robot_body_ang_vel_w=b["ang_vel"].numpy(),
        ref_body_pos_w=ref.body_pos_w.numpy(),
        ref_body_quat_w=ref.body_quat_w.numpy(),
        ref_body_lin_vel_w=ref.body_lin_vel_w.numpy(),
        ref_body_ang_vel_w=ref.body_ang_vel_w.numpy(),
        anchor_index=env.anchor_index,
        action=env.action,
        prev_action=env.prev_action,
        joint_pos=env.sm_state()[:, 7:36],
        soft_limits=env.soft_joint_limits,
        cfg=H2SuperMimicCfg().reward,
        dt=0.02,
    )
    for k, v in n_terms.items():
        np.testing.assert_allclose(t_terms[k].numpy(), v, atol=1e-6, err_msg=k)
    # Zero-initialised dynamics = constant-velocity integration.
    with torch.no_grad():
        s1 = wm.step(s, a, xi)
    np.testing.assert_allclose(
        s1[:, 7:36].numpy(), (s[:, 7:36] + 0.02 * s[:, 42:]).numpy(), atol=1e-6
    )


def test_clip_end_truncates(clips):
    env = _make_env(
        clips,
        num_envs=1,
        sampling_mode="start",
        max_episode_seconds=None,
        reset_noise={"enabled": False},
        domain_rand={"enabled": False},
    )
    lib = env.library
    env.forced_frame[:] = lib.clip_start[1]
    env.init_state()
    steps, done = 0, False
    while not done and steps < 500:
        st = env.step(np.zeros((1, 29), np.float32))
        steps += 1
        done = bool(st.terminated[0] or st.truncated[0])
    assert done
    if not st.terminated[0]:
        # Reset at f -> targets f+1 .. last, plus one step at the clamped last frame.
        assert steps == lib.clip_len[1]


@pytest.mark.slow
def test_runner_smoke(clips, tmp_path):
    from unilab.algos.torch.supermimic import SuperMimicRunner

    env = _make_env(clips, num_envs=8)
    cfg = _small_cfg()
    cfg.replay_capacity = 64
    cfg.wm_trainer.batch_size = 32
    cfg.wm_trainer.batches_per_update = 1
    cfg.policy_trainer.batch_size = 16
    cfg.policy_trainer.horizon = 4
    cfg.schedule.total_iterations = 60
    cfg.schedule.warmup_iterations = 20
    cfg.schedule.log_interval = 30
    cfg.schedule.checkpoint_interval = 60
    runner = SuperMimicRunner(env, cfg, "cpu", tmp_path)
    runner.run()
    assert runner.num_updates > 0
    assert (tmp_path / "ckpt_latest.pt").exists()
