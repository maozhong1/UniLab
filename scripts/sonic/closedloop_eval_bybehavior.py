"""Per-behavior closed-loop discriminator for the WARM INIT (pure ``last.pt``).

Same idea as ``closedloop_eval.py`` (run last.pt as a DETERMINISTIC closed-loop
controller inside the ``G1SonicMotionTracking`` UniLab env, action_output remap ON,
no DR, no exploration) BUT:

  * feeds the FULL bones_seed_500 clip set (the exact motions the warm finetune used),
  * assigns a random clip per env from frame 0 (sampling_mode="start"),
  * snapshots each env's initial clip -> behavior label (walk/run/jump/dance/idle),
  * reports survival + min pelvis height + joint RMSE BROKEN DOWN BY BEHAVIOR.

Verdict logic per behavior:
  survived-full & upright  -> env is byte/convention compatible for that behavior;
                              a bad finetune there is a TRAINING-METHOD issue.
  falls fast (low pelvis)  -> obs/action/physics assembly not deploy-compatible for
                              that behavior; finetune optimizes an untransferable target.

Run:
    HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python \
        scripts/sonic/closedloop_eval_bybehavior.py \
        --ckpt /home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/sonic_release/last.pt \
        --motion_dir /home/maozhong/work/sonic_vla_infer/bones_seed_500/npz \
        --num-envs 100 --steps 300
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from unilab.algos.torch.sonic.core import SonicG1Core, load_g1_from_last_pt
from unilab.base import registry

registry.ensure_registries()

ENC = SonicG1Core.ENC_INPUT_DIM      # 640
PROPRIO = SonicG1Core.PROPRIO_DIM    # 930


def behavior_of(name: str) -> str:
    n = os.path.basename(name).lower()
    if "dance" in n:
        return "dance"
    if "jump" in n:
        return "jump"
    if "jog" in n or "run" in n or "skip" in n:
        return "run"
    if "walk" in n:
        return "walk"
    if "idle" in n or "rotate" in n:
        return "idle"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/sonic_release/last.pt",
    )
    ap.add_argument("--task", default="G1SonicMotionTracking")
    ap.add_argument("--sim", default="mujoco")
    ap.add_argument("--motion_dir", required=True, help="dir of NPZ clips (the finetune set).")
    ap.add_argument("--num-envs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=300, help="rollout horizon (steps).")
    ap.add_argument("--fall-height", type=float, default=0.4,
                    help="pelvis height (m) below which we call it fallen.")
    ap.add_argument("--strict-height", type=float, default=0.35,
                    help="SONIC strict tracking-error termination threshold (m). "
                         "Default 0.35 (loosened from the env default 0.15).")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    files = sorted(glob.glob(os.path.join(a.motion_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz under {a.motion_dir}")
    print(f"Loaded {len(files)} clips from {a.motion_dir}")

    # "clip_start" = each env starts at frame 0 of a RANDOMLY chosen clip.
    # (NOTE: "start" means global frame 0 = clip 0 only -> would test one clip.)
    override: dict = {
        "sampling_mode": "clip_start",
        "action_output_isaaclab_to_mujoco": True,
        "strict_height_threshold": a.strict_height,
        "motion_file": files,
    }
    env = registry.make(a.task, num_envs=a.num_envs, sim_backend=a.sim,
                        env_cfg_override=override)

    print(f"strict_height_threshold in effect: {getattr(env._cfg, 'strict_height_threshold', '?')} m")

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
            return core.act_mean(enc, proprio).numpy().astype(np.float32)

    try:
        state = env.init_state()
        # snapshot each env's INITIAL clip -> behavior (before any resets).
        # Derive clip from the absolute frame (current_clip_indices is not maintained
        # in clip_start mode); clip i owns frames [clip_offsets[i], clip_offsets[i+1]).
        cf0 = np.asarray(env.motion_sampler.current_frames, dtype=np.int64)
        co = np.asarray(env.motion_loader.clip_offsets, dtype=np.int64)
        init_clip = np.searchsorted(co, cf0, side="right") - 1
        env_behavior = np.array([behavior_of(files[i]) for i in init_clip])

        first_term = np.full(N, -1, dtype=np.int64)
        min_h = np.full(N, np.inf)
        rmse_acc = np.zeros(N)
        rmse_cnt = np.zeros(N)

        for t in range(a.steps):
            action = act(np.asarray(state.obs["obs"]))
            state = env.step(action)

            rb_pos, _ = env._backend.get_body_pose_w(env.body_ids)
            h = np.asarray(rb_pos[:, ai, 2], dtype=np.float64)
            min_h = np.minimum(min_h, h)

            try:
                frames = env.motion_sampler.current_frames.copy()
                md = env.motion_loader.get_motion_at_frame(frames)
                ref_q = np.asarray(md.joint_pos, dtype=np.float64)
                robot_q = np.asarray(env.get_dof_pos(), dtype=np.float64)
                per_env = np.sqrt(np.mean((robot_q - ref_q) ** 2, axis=1))
                alive = first_term < 0
                rmse_acc[alive] += per_env[alive]
                rmse_cnt[alive] += 1
            except Exception:
                pass

            term = np.asarray(state.terminated).reshape(-1).astype(bool)
            fell = term | (h < a.fall_height)
            newly = fell & (first_term < 0)
            first_term[newly] = t

        horizon = a.steps
        survived = first_term < 0
        surv_steps = np.where(survived, horizon, first_term).astype(float)
        rmse = np.where(rmse_cnt > 0, rmse_acc / np.maximum(rmse_cnt, 1), np.nan)

        print("\n================ PER-BEHAVIOR CLOSED-LOOP DISCRIMINATOR ================")
        print(f"num_envs={N}  horizon={horizon} steps  fall_height={a.fall_height} m  "
              f"(policy=deterministic act_mean, DR off, action remap ON)")
        print(f"{'behavior':10s} {'n':>4s} {'surv_full':>10s} {'mean_steps':>11s} "
              f"{'min_h(m)':>9s} {'rmse(rad)':>10s}")
        order = ["walk", "run", "jump", "dance", "idle", "other"]
        present = [b for b in order if (env_behavior == b).any()]
        for b in present:
            m = env_behavior == b
            n = int(m.sum())
            sf = float((survived & m).sum()) / n
            ms = float(surv_steps[m].mean())
            mh = float(np.nanmin(min_h[m]))
            rr = float(np.nanmean(rmse[m]))
            print(f"{b:10s} {n:4d} {sf*100:9.0f}% {ms:11.1f} {mh:9.3f} {rr:10.4f}")
        # overall
        n = N
        sf = float(survived.sum()) / n
        print("-" * 60)
        print(f"{'ALL':10s} {n:4d} {sf*100:9.0f}% {surv_steps.mean():11.1f} "
              f"{float(np.nanmin(min_h)):9.3f} {float(np.nanmean(rmse)):10.4f}")

        print("\nInterpretation:")
        print("  surv_full high + min_h>~0.5  -> env compatible for that behavior "
              "(finetune failure = training method).")
        print("  surv_full low  + min_h<0.4   -> warm policy FALLS here "
              "-> obs/action/physics mismatch for that behavior.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
