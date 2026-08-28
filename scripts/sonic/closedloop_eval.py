"""Discriminator: is the UniLab env deploy-compatible? Run the WARM INIT (pure
``last.pt``) as a CLOSED-LOOP controller in ``G1SonicMotionTracking`` with
DETERMINISTIC actions (``core.act_mean``, no exploration) and no domain randomization,
and measure how long it stays upright while tracking the reference.

Rationale: ``last.pt`` stands/walks/dances STABLY in the MuJoCo simulator and the deploy
C++. The UniLab env is also a MuJoCo backend. So if the SAME weights, driven by the
env's OWN obs, fall over quickly here, the env's obs/action assembly is NOT byte/
convention-compatible with deploy -> any finetune trained in this env optimizes toward a
target that does not transfer to deploy (that would be the real blocker, not the number
of iterations). If instead it survives the full horizon, the env is compatible and the
finetune failure is purely a training-method issue (cold-critic warm-start collapse +
undertraining).

This is CLOSED-LOOP (policy in the loop), unlike ``openloop_replay.py`` which only drives
PD targets to the reference and never exercises the policy's balance.

Run:
    HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python scripts/sonic/closedloop_eval.py \
        --ckpt ./last.pt --num-envs 8 --steps 500
    # optionally track the same clips the finetune used:
    #   --motion_file /home/maozhong/work/sonic_vla_infer/bones_seed_subset/npz/<clip>.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from unilab.algos.torch.sonic.core import SonicG1Core, load_g1_from_last_pt
from unilab.base import registry

registry.ensure_registries()

ENC = SonicG1Core.ENC_INPUT_DIM      # 640
PROPRIO = SonicG1Core.PROPRIO_DIM    # 930


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        # Relative to the launch dir (repo root under `uv run python scripts/...`).
        default="./last.pt",
        help="sonic last.pt (the warm init) to evaluate closed-loop; default ./last.pt.",
    )
    ap.add_argument("--task", default="G1SonicMotionTracking")
    ap.add_argument("--sim", default="mujoco")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500, help="rollout horizon (steps).")
    ap.add_argument("--motion_file", default=None, help="override env.motion_file (NPZ).")
    ap.add_argument("--fall-height", type=float, default=0.4,
                    help="pelvis height (m) below which we call it fallen.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    # deterministic start (frame 0, follow consecutively); DR toggles already default off.
    override: dict = {
        "sampling_mode": "start",
        "action_output_isaaclab_to_mujoco": True,
    }
    if a.motion_file:
        override["motion_file"] = a.motion_file
    env = registry.make(a.task, num_envs=a.num_envs, sim_backend=a.sim,
                        env_cfg_override=override)

    core = SonicG1Core(with_kin_aux=False, use_fsq=True).eval()
    load_g1_from_last_pt(core, a.ckpt)
    print(f"Loaded warm init: {a.ckpt}")
    print(f"FSQ backend: {'official' if core.fsq_is_official else 'FSQFallback'}")

    N = a.num_envs
    ai = env.anchor_body_idx

    def act(obs_obs: np.ndarray) -> np.ndarray:
        enc = torch.from_numpy(obs_obs[:, :ENC].astype(np.float32))
        proprio = torch.from_numpy(obs_obs[:, ENC:ENC + PROPRIO].astype(np.float32))
        with torch.no_grad():
            return core.act_mean(enc, proprio).numpy().astype(np.float32)  # deterministic

    try:
        state = env.init_state()
        first_term = np.full(N, -1, dtype=np.int64)   # step of first fall/termination (-1 = never)
        min_h = np.full(N, np.inf)
        joint_rmse = []

        for t in range(a.steps):
            action = act(np.asarray(state.obs["obs"]))
            state = env.step(action)

            # pelvis (anchor) height as a fall proxy
            rb_pos, _ = env._backend.get_body_pose_w(env.body_ids)
            h = np.asarray(rb_pos[:, ai, 2], dtype=np.float64)
            min_h = np.minimum(min_h, h)

            # tracking quality vs the reference the env is commanding
            try:
                frames = env.motion_sampler.current_frames.copy()
                md = env.motion_loader.get_motion_at_frame(frames)
                ref_q = np.asarray(md.joint_pos, dtype=np.float64)   # (N, n)
                robot_q = np.asarray(env.get_dof_pos(), dtype=np.float64)
                joint_rmse.append(float(np.sqrt(np.mean((robot_q - ref_q) ** 2))))
            except Exception:
                pass

            term = np.asarray(state.terminated).reshape(-1).astype(bool)
            fell = term | (h < a.fall_height)
            newly = fell & (first_term < 0)
            first_term[newly] = t

        horizon = a.steps
        survived = first_term < 0
        surv_steps = np.where(survived, horizon, first_term)
        print("\n================ CLOSED-LOOP DISCRIMINATOR ================")
        print(f"num_envs={N}  horizon={horizon} steps  fall_height={a.fall_height} m")
        print(f"per-env steps-until-fall: {surv_steps.tolist()}")
        print(f"survived full horizon:    {int(survived.sum())}/{N} envs")
        print(f"mean steps-until-fall:    {surv_steps.mean():.1f} / {horizon}")
        print(f"min pelvis height (m):    per-env {np.round(min_h,3).tolist()}")
        if joint_rmse:
            jr = np.asarray(joint_rmse)
            print(f"joint tracking RMSE (rad): mean={jr.mean():.4f}  max={jr.max():.4f}")

        frac = survived.mean()
        if frac >= 0.75:
            verdict = ("ENV COMPATIBLE -> last.pt is a stable closed-loop controller here. "
                       "Finetune failure = TRAINING METHOD (cold-critic collapse + undertrain).")
        elif frac <= 0.25:
            verdict = ("ENV NOT DEPLOY-COMPATIBLE -> deploy-stable last.pt falls fast here. "
                       "Fix obs/action byte-compat BEFORE any finetune; iterations won't help.")
        else:
            verdict = ("MIXED -> partial stability; inspect obs/action assembly and reset noise.")
        print(f"\nVERDICT: {verdict}")
        print("(policy=deterministic act_mean, no exploration; DR toggles default off.)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
