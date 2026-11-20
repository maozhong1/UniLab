"""Benchmark a SuperMimic H2 policy (training checkpoint or ONNX) on reference clips.

Usage:
    uv run scripts/supermimic/eval_benchmark.py --onnx h2_mimic_full.onnx --npz-dir <dir> [--npz-dir <dir2>] \
        [--clip-list names.txt | --sample 400] [--num-envs 256] [--out results.json]
    uv run scripts/supermimic/eval_benchmark.py --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt --npz-dir <dir>
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from unilab.algos.torch.supermimic.benchmark import (
    load_policy_from_checkpoint,
    onnx_policy_fn,
    run_benchmark,
    save_results,
    summarize,
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
    p.add_argument(
        "--clip-list",
        type=str,
        default=None,
        help="text file with one clip name (npz stem) per line",
    )
    p.add_argument(
        "--sample", type=int, default=0, help="evaluate a random subset of this many clips (seed 0)"
    )
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    files = discover_motion_files((), args.npz_dir)
    if args.clip_list:
        wanted = {ln.strip() for ln in Path(args.clip_list).read_text().splitlines() if ln.strip()}
        files = [f for f in files if Path(f).stem in wanted]
    if args.sample:
        files = sorted(random.Random(0).sample(files, min(args.sample, len(files))))
    print(f"[eval] {len(files)} clips")
    if args.onnx:
        policy_fn = onnx_policy_fn(args.onnx)
    else:
        policy_fn = torch_policy_fn(
            load_policy_from_checkpoint(args.checkpoint, args.device), args.device
        )

    registry.ensure_registries()

    def make_env(num_envs, overrides):
        return registry.make(
            "H2SuperMimic", sim_backend="mujoco", num_envs=num_envs, env_cfg_override=overrides
        )

    results = run_benchmark(make_env, files, policy_fn, num_envs=args.num_envs)
    s = summarize(results)
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    print(
        f"[eval] clips {s['clips']} | failed {s['failed']} | mean norm return {s['mean_norm_return']:.3f} | step-weighted {s['step_weighted_norm_return']:.3f}"
    )
    for r in worst:
        print(f"    {r.norm_return:.3f} {'FAIL' if r.failed else '    '} {r.name}")
    if args.out:
        save_results(results, args.out)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
