"""Record deterministic H2 SuperMimic rollouts (robot qpos + reference qpos per control step) for rendering.

Usage:
    uv run scripts/supermimic/record_rollout.py --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt \
        --npz-dir <dir> [--clips <stem> ...] --out media/h2_rollouts.npz

Same protocol as eval_benchmark.py: every clip starts at its first frame, no observation / action noise,
no reset noise, no domain randomisation. Each clip is played once; recording stops at the clip end or at
the termination. Render with scripts/supermimic/render_rollout.py. Output layout follows the reference
SuperMimic record_rollout.py (names, robot_qpos, ref_qpos, lengths, norm_return).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from unilab.algos.torch.supermimic.benchmark import (
    EVAL_ENV_OVERRIDES,
    load_policy_from_checkpoint,
    onnx_policy_fn,
    torch_policy_fn,
)
from unilab.base import registry
from unilab.envs.motion_tracking.h2.supermimic_library import discover_motion_files


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str)
    src.add_argument("--onnx", type=str)
    p.add_argument("--npz-dir", action="append", required=True)
    p.add_argument("--clips", nargs="*", default=None, help="npz stems (default: all in --npz-dir)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--out", type=str, default="media/h2_rollouts.npz")
    args = p.parse_args()

    files = discover_motion_files((), args.npz_dir)
    if args.clips:
        missing = set(args.clips) - {Path(f).stem for f in files}
        assert not missing, f"clips not found: {sorted(missing)}"
        files = [f for f in files if Path(f).stem in set(args.clips)]
    if args.onnx:
        policy_fn = onnx_policy_fn(args.onnx)
    else:
        policy_fn = torch_policy_fn(
            load_policy_from_checkpoint(args.checkpoint, args.device), args.device
        )

    registry.ensure_registries()
    n = len(files)
    env = registry.make(
        "H2SuperMimic",
        sim_backend="mujoco",
        num_envs=n,
        env_cfg_override={**EVAL_ENV_OVERRIDES, "motion_file": list(files), "motion_dirs": []},
    )
    try:
        lib = env.library
        nq = 7 + lib.default_joint_pos.shape[0]
        ref_states = lib.ref_states()
        max_r = env.cfg.reward.max_weight_sum * env.cfg.ctrl_dt
        clip_ids = np.arange(n)
        env.forced_frame[:] = lib.clip_start[clip_ids]
        state = env.init_state()
        env.forced_frame[:] = lib.clip_start[clip_ids]  # keep the clip if an env is reset again

        rob, ref = [], []
        alive = np.ones(n, dtype=bool)
        ret_sum = np.zeros(n)
        steps = np.zeros(n, dtype=np.int64)
        failed = np.zeros(n, dtype=bool)

        def snapshot() -> None:
            rob.append(env.sm_state()[:, :nq].copy())
            ref.append(ref_states[env.frame, :nq].copy())

        snapshot()
        while alive.any() and steps.max() < args.max_steps:
            state = env.step(policy_fn(state.obs["obs"], state.obs["goal"]).astype(np.float32))
            ret_sum += state.reward * alive
            steps += alive
            failed |= state.terminated & alive
            alive &= ~(state.terminated | state.truncated)
            if alive.any():
                snapshot()
    finally:
        env.close()

    T = len(rob)
    robot_qpos, ref_qpos = np.stack(rob, 1), np.stack(ref, 1)
    for i in range(n):  # freeze frames after a clip's end so every row has the same length
        robot_qpos[i, steps[i] :] = robot_qpos[i, steps[i] - 1]
        ref_qpos[i, steps[i] :] = ref_qpos[i, steps[i] - 1]
    names = [lib.clip_names[i] for i in range(n)]
    norm_return = ret_sum / (max_r * np.maximum(steps, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        robot="h2",
        names=np.array(names),
        robot_qpos=robot_qpos[:, :T],
        ref_qpos=ref_qpos[:, :T],
        lengths=steps,
        norm_return=norm_return,
        failed=failed,
    )
    for i, nm in enumerate(names):
        print(
            f"{nm:56s} steps {int(steps[i]):5d}/{int(lib.clip_len[i])}  return {norm_return[i]:.3f}"
            f"{'  FAIL' if failed[i] else ''}"
        )
    print("saved", out)


if __name__ == "__main__":
    main()
