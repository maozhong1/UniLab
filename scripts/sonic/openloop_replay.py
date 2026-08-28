"""Task-4 dynamics alignment: headless open-loop control replay of the sonic env.

Validates joint order / sign / PD gains / action_scale / default offset for the
G1SonicMotionTracking env by driving the REAL control pipeline open-loop: at each
step the action is set so the PD position target equals the reference joint pose
(``action = (ref_joint_pos - default) / action_scale``, the inverse of
``apply_action``), then joint / anchor tracking error is measured.

If the joint mapping or a sign is wrong, position-target tracking of a feasible
reference diverges immediately (robot falls) and the error blows up — the same
robocasa-style divergence test used for deploy debugging. On a correctly aligned
model the per-joint error stays small and the robot stays upright.

Run (headless, uses bundled dance1 by default):
    uv run --no-sync python scripts/sonic/openloop_replay.py
    uv run --no-sync python scripts/sonic/openloop_replay.py --motion_file <bones_seed.npz> --steps 300
"""
from __future__ import annotations

import argparse

import numpy as np

from unilab.base import registry

registry.ensure_registries()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="G1SonicMotionTracking")
    ap.add_argument("--motion_file", default=None, help="override env.motion_file (NPZ)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sim", default="mujoco")
    a = ap.parse_args()

    override: dict = {"sampling_mode": "start"}  # start at frame 0, follow consecutively
    if a.motion_file:
        override["motion_file"] = a.motion_file
    env = registry.make(a.task, num_envs=1, sim_backend=a.sim, env_cfg_override=override)

    try:
        n = env._num_action
        action_scale = float(np.asarray(env._cfg.control_config.action_scale).reshape(-1)[0])
        default = np.asarray(env.default_angles, dtype=np.float64).reshape(-1)
        ai = env.anchor_body_idx

        env.init_state()

        joint_rmse = []
        anchor_pos_err = []
        upright = []
        for _ in range(a.steps):
            frames = env.motion_sampler.current_frames.copy()
            md = env.motion_loader.get_motion_at_frame(frames)
            ref_q = np.asarray(md.joint_pos[0], dtype=np.float64)  # (n,)
            ref_anchor_pos = np.asarray(md.body_pos_w[0, ai], dtype=np.float64)

            # inverse of apply_action so PD target == ref joint pose
            action = ((ref_q - default) / action_scale).astype(np.float32)[None, :]
            state = env.step(action)

            robot_q = np.asarray(env.get_dof_pos()[0], dtype=np.float64)
            joint_rmse.append(float(np.sqrt(np.mean((robot_q - ref_q) ** 2))))

            rb_pos, _ = env._backend.get_body_pose_w(env.body_ids)
            robot_anchor_pos = np.asarray(rb_pos[0, ai], dtype=np.float64)
            anchor_pos_err.append(float(np.linalg.norm(robot_anchor_pos - ref_anchor_pos)))
            upright.append(float(robot_anchor_pos[2]))  # pelvis height proxy

            if state.terminated[0] or state.truncated[0]:
                # env auto-resets; keep going to see if it re-tracks
                pass

        jr = np.asarray(joint_rmse)
        ap_ = np.asarray(anchor_pos_err)
        print(f"steps={a.steps} action_scale={action_scale} n_dof={n} anchor_idx={ai}")
        print(f"joint RMSE (rad):   mean={jr.mean():.4f}  p95={np.percentile(jr,95):.4f}  max={jr.max():.4f}")
        print(f"anchor pos err (m): mean={ap_.mean():.4f}  p95={np.percentile(ap_,95):.4f}  max={ap_.max():.4f}")
        print(f"pelvis height (m):  first={upright[0]:.3f}  last={upright[-1]:.3f}  min={min(upright):.3f}")

        # Divergence heuristics: a correctly aligned model tracks position targets
        # with small steady error and stays upright; a wrong sign/order explodes.
        diverged = bool(jr.max() > 1.5 or min(upright) < 0.3)
        verdict = "DIVERGED (check joint order/sign/gear/PD)" if diverged else "STABLE (alignment plausible)"
        print(f"\nVERDICT: {verdict}")
        print("Note: 'STABLE' means the sim tracked open-loop position targets; for full")
        print("last.pt-compat validation also diff obs assembly vs sonic (see mapping doc).")
    finally:
        env.close()


if __name__ == "__main__":
    main()
