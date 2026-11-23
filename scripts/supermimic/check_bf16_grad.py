"""Check that bf16 MLP matmuls leave the BPTT policy gradient intact.

Loads a training checkpoint, collects a little replay data with its policy (training noise / DR),
then computes the imagined-rollout policy gradient (Eq. 2 objective, critic bootstrap) on the
same start batch in float32 and in bfloat16, for several horizons. The bf16 error is compared
with the gradient's own sampling noise (float32 on a different start batch), which is the noise
level training already works with.

Runs on the CPU at low priority so a training run on the XPU is not disturbed:
    OMP_NUM_THREADS=2 nice -n 19 uv run --no-sync python scripts/supermimic/check_bf16_grad.py \
        --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_0003000.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import torch

from unilab.algos.torch.supermimic import SuperMimicCfg, SuperMimicRunner
from unilab.algos.torch.supermimic.imagination import (
    ImagStep,
    imagine,
    obs_noise_half,
    policy_objective,
)
from unilab.algos.torch.supermimic.world_model import set_mlp_autocast
from unilab.base import registry
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.motion_tracking.h2.supermimic_library import discover_motion_files


def policy_grad(
    runner: SuperMimicRunner, start, horizon: int, dtype: torch.dtype | None, noise_seed: int
):
    wm, policy, critic, lib = runner.wm, runner.policy, runner.critic, runner.lib
    for m in (wm, policy, critic):
        set_mlp_autocast(m, dtype)
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    pcfg = runner.cfg.policy_trainer
    noise = (
        obs_noise_half(lib.default_joint_pos.numel(), runner.env.cfg.obs_noise_half)
        if pcfg.obs_noise
        else None
    )
    torch.manual_seed(noise_seed)
    ro = imagine(
        wm,
        policy,
        lib,
        start,
        horizon,
        policy.cfg.future_frames,
        ImagStep(wm, lib.default_joint_pos),
        noise,
    )
    H1, n, _ = ro.states.shape
    feat = wm.spec.canonical(ro.states[-1])
    v_H = critic(feat, ro.goals[-1], wm.spec.normalize_xi(start.xi), ro.e_root[-1])
    J = policy_objective(ro, v_H, pcfg.gamma)
    policy.zero_grad(set_to_none=True)
    (-J / horizon).backward()
    g = torch.cat([p.grad.reshape(-1) for p in policy.parameters()])
    per_module = {
        name: torch.cat([p.grad.reshape(-1) for p in mod.parameters()])
        for name, mod in (("encoder", policy.encoders["motion"]), ("decoder", policy.decoder))
    }
    imag_ret = float((ro.survival[:-1] * ro.rewards).sum(0).mean())
    for m in (wm, policy, critic):
        set_mlp_autocast(m, None)
    return g, per_module, float(J), imag_ret


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b||"""
    return float((a - b).norm() / b.norm())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--npz-dir", action="append", default=None)
    p.add_argument("--clips", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--collect-steps", type=int, default=120)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--repeats", type=int, default=3, help="independent start-batch pairs")
    p.add_argument("--out", default="/tmp/supermimic_bf16_grad.json")
    args = p.parse_args()
    t0 = time.time()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    dirs = args.npz_dir or [
        str(
            Path(
                "~/work/sonic_vla_infer/bones-seed/selections/upright_nzp_h2_29dof_new"
            ).expanduser()
        ),
        str(Path("~/work/sonic_vla_infer/bones_seed_3k_new/npz_h2_29dof_new").expanduser()),
    ]
    files = sorted(random.Random(0).sample(discover_motion_files((), dirs), args.clips))
    registry.ensure_registries()
    env = registry.make(
        "H2SuperMimic",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override={"motion_file": files, "seed": 0},
    )

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = SuperMimicCfg()
    apply_cfg_overrides(cfg, ck["config"])
    cfg.mlp_dtype = "float32"
    cfg.wm_trainer.compile = None
    cfg.policy_trainer.compile = None
    cfg.replay_capacity = args.collect_steps + 20
    it0 = int(ck["iteration"])
    sch = cfg.schedule
    sch.total_iterations = it0 + args.collect_steps
    sch.warmup_iterations = 10**9  # collect only, no updates
    sch.log_interval = sch.checkpoint_interval = 10**9
    runner = SuperMimicRunner(
        env, cfg, "cpu", Path("/tmp/supermimic_bf16_grad_run"), resume=args.checkpoint
    )
    runner.save = lambda *a, **k: None  # type: ignore[method-assign]
    runner.run()
    print(
        f"[bf16] collected {runner.buffer.size} replay rows in {time.time() - t0:.0f}s", flush=True
    )

    results = []
    for h in args.horizons:
        for r in range(args.repeats):
            torch.manual_seed(1000 + r)
            start_a = runner.buffer.sample_starts(args.batch, cfg.policy.history)
            start_b = runner.buffer.sample_starts(args.batch, cfg.policy.history)
            g32, m32, J32, ret32 = policy_grad(runner, start_a, h, None, noise_seed=r)
            g32_again, *_ = policy_grad(runner, start_a, h, None, noise_seed=r)
            g16, m16, J16, ret16 = policy_grad(runner, start_a, h, torch.bfloat16, noise_seed=r)
            g32_b, *_ = policy_grad(runner, start_b, h, None, noise_seed=r + 100)
            row = {
                "horizon": h,
                "repeat": r,
                "fp32_repeat_rel_err": rel(g32_again, g32),
                "bf16_cos": cos(g16, g32),
                "bf16_rel_err": rel(g16, g32),
                "bf16_norm_ratio": float(g16.norm() / g32.norm()),
                "bf16_cos_encoder": cos(m16["encoder"], m32["encoder"]),
                "bf16_cos_decoder": cos(m16["decoder"], m32["decoder"]),
                "noise_cos": cos(g32_b, g32),
                "noise_rel_err": rel(g32_b, g32),
                "J_fp32": J32,
                "J_bf16": J16,
                "imag_ret_fp32": ret32,
                "imag_ret_bf16": ret16,
            }
            results.append(row)
            print(
                f"[bf16] H={h:2d} rep {r}: bf16 cos {row['bf16_cos']:.4f} rel {row['bf16_rel_err']:.3f} norm x{row['bf16_norm_ratio']:.3f} "
                f"(enc {row['bf16_cos_encoder']:.4f} dec {row['bf16_cos_decoder']:.4f}) | sampling noise cos {row['noise_cos']:.4f} rel {row['noise_rel_err']:.3f} | "
                f"imag_ret {ret32:.4f} vs {ret16:.4f} | fp32 repeat rel {row['fp32_repeat_rel_err']:.1e} | {time.time() - t0:.0f}s",
                flush=True,
            )
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"[bf16] wrote {args.out}")


if __name__ == "__main__":
    main()
