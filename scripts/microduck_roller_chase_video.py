"""Chase-camera video for MicroduckRollers checkpoints.

Spawn yaw is random, and the stock tracking camera keeps a world-fixed azimuth,
so a duck can skate straight at the camera. Here each duck gets its own camera
whose azimuth follows the (smoothed) base yaw every frame; with --num-envs > 1
the views are tiled in a grid. By default every duck gets a fixed forward push
(--cmd-x); --random-cmd keeps the env's own sampled commands.

Usage:
  uv run --extra mujoco --extra microduck --no-sync python scripts/microduck_roller_chase_video.py \
      --ckpt logs/rsl_rl_ppo/MicroduckRollers/<run>/model_<N>.pt \
      --num-envs 4 --steps 800 --cmd-x 0.5 --out roller_chase.mp4
"""

import argparse
import glob
import os
import tempfile
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
import torch
from hydra import compose, initialize_config_dir

import unilab.envs.locomotion.microduck  # noqa: F401  (registers MicroduckRollers)
from unilab.base.backend import materialize_scene_visual_override
from unilab.base.backend.mujoco.playback import resolve_render_play_model_files
from unilab.training import create_env
from unilab.training.backend_adapter import BackendAdapter

ROOT = Path(__file__).resolve().parents[1]
CMD_X_OBS_IDX = 48  # gyro3 + grav3 + diff14 + dof_vel14 + last_act14 → twist starts at 48


def latest_ckpt() -> str:
    runs = sorted(glob.glob(str(ROOT / "logs/rsl_rl_ppo/MicroduckRollers/*_mujoco")))
    ckpts = glob.glob(os.path.join(runs[-1], "model_*.pt"))
    return max(ckpts, key=lambda p: int(Path(p).stem.split("_")[1]))


def load_actor(path: str) -> torch.nn.Module:
    sd = torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"]
    mlp = torch.nn.Sequential(
        torch.nn.Linear(61, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 14),
    )
    mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
    return mlp.eval()


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num-envs", type=int, default=1, help="ducks; >1 tiles one chase view each")
    ap.add_argument("--cmd-x", type=float, default=0.5, help="push command, (0, 0.6] = forward")
    ap.add_argument("--random-cmd", action="store_true",
                    help="keep the env's own sampled/resampled commands instead of --cmd-x")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--width", type=int, default=768, help="full output width")
    ap.add_argument("--height", type=int, default=480, help="full output height")
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    # The trunk faces body +x; a free camera at azimuth=yaw sits behind it.
    ap.add_argument("--behind", type=float, default=0.0,
                    help="azimuth offset from heading (deg): 0=behind, 180=front, 90=side")
    ap.add_argument("--yaw-smooth", type=float, default=0.05, help="EMA factor for camera yaw")
    ap.add_argument("--out", default="roller_chase.mp4")
    args = ap.parse_args()

    n = args.num_envs
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    tile_w, tile_h = args.width // cols, args.height // rows

    ckpt = args.ckpt or latest_ckpt()
    print(f"checkpoint: {ckpt}")

    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_rollers/mujoco"])
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo", scene_materializer=materialize_scene_visual_override
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=n, env_cfg_override=override)
    actor = load_actor(ckpt)

    def pin_cmd() -> None:
        # Re-pin every step so resamples / auto-resets keep the forward push.
        if args.random_cmd:
            return
        cmd = env.state.info["commands"]
        cmd[:, :] = 0.0
        cmd[:, 0] = args.cmd_x

    env.init_state()
    pin_cmd()
    obs = env.state.obs["obs"].copy()
    if not args.random_cmd:
        obs[:, CMD_X_OBS_IDX] = args.cmd_x

    # Same visual model resolution as the eval/record path (physics model has no
    # textures/meshes for rendering).
    with tempfile.TemporaryDirectory(prefix="duck-chase-") as tmp:
        files = resolve_render_play_model_files(env, num_envs=1, tmp_dir=tmp)
        path = files[0] if isinstance(files, list) else files
        model = (mujoco.MjModel.from_binary_path if str(path).endswith(".mjb")
                 else mujoco.MjModel.from_xml_path)(str(path))
    snap_len = np.asarray(env.get_physics_state_snapshot()).shape[-1]
    assert snap_len == 1 + model.nq + model.nv, (snap_len, model.nq, model.nv)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, tile_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, tile_h)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=tile_h, width=tile_w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.distance
    cam.elevation = args.elevation

    fps = int(round(1.0 / float(env.cfg.ctrl_dt)))
    writer = iio.get_writer(args.out, fps=fps, macro_block_size=16)
    cam_yaw: list[float | None] = [None] * n
    vx_log = []
    with torch.inference_mode():
        for t in range(args.steps):
            act = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            st = env.step(act)
            pin_cmd()
            obs = st.obs["obs"]
            vx_log.append(np.asarray(st.obs["critic"][:, -3]))

            snap = np.asarray(env.get_physics_state_snapshot())
            frame = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
            for i in range(n):
                s = snap[i]
                data.qpos[:] = s[1 : 1 + model.nq]
                data.qvel[:] = s[1 + model.nq : 1 + model.nq + model.nv]
                mujoco.mj_forward(model, data)
                yaw = quat_yaw(data.qpos[3:7])
                prev = cam_yaw[i]
                # EMA on the wrapped angle difference
                cam_yaw[i] = yaw if prev is None else prev + args.yaw_smooth * float(
                    np.angle(np.exp(1j * (yaw - prev))))
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.08]
                cam.azimuth = np.degrees(cam_yaw[i]) + args.behind
                renderer.update_scene(data, camera=cam)
                r, c = divmod(i, cols)
                frame[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = renderer.render()
            writer.append_data(frame)
            if (t + 1) % 200 == 0:
                v = np.mean(vx_log[-200:], axis=0)
                print(f"step {t + 1}/{args.steps}  mean body vx(last 200) per duck={np.round(v, 3)} m/s")
    writer.close()
    renderer.close()
    print(f"saved {args.out}  ({args.steps} frames @ {fps} fps, {cols * tile_w}x{rows * tile_h}, {n} ducks)")


if __name__ == "__main__":
    main()
