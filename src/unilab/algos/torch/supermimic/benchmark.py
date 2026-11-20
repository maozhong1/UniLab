"""Benchmark evaluation (SuperMimic Appendix F.1).

Every clip is played once from its first frame with the deterministic policy: no observation /
action noise, no reset noise, no domain randomisation (foot friction 1.0). A clip *fails* if the
training termination (pelvis height / tilt versus the reference) fires before the clip ends.
Normalised return = sum of step rewards / (max step reward * steps).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .nets import Policy, PolicyCfg

PolicyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (obs_hist, goal) -> action

EVAL_ENV_OVERRIDES: dict[str, Any] = {
    "max_episode_seconds": None,  # the clip end decides
    "action_noise_std": 0.0,
    "obs_noise": False,
    "reset_noise": {"enabled": False},
    "domain_rand": {"enabled": False},
    "sampling_mode": "start",
    "motion_exclude_keywords": [],
    "seed": 0,
}


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    terms: dict


def load_policy_from_checkpoint(path: str | Path, device: str) -> Policy:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["policy"]
    policy = Policy(
        PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    ).to(device)
    policy.load_state_dict(ck["policy"])
    return policy.eval()


def torch_policy_fn(policy: Policy, device: str) -> PolicyFn:
    @torch.no_grad()
    def fn(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return (
            policy(torch.as_tensor(obs, device=device), torch.as_tensor(goal, device=device))
            .cpu()
            .numpy()
        )

    return fn


def onnx_policy_fn(path: str | Path) -> PolicyFn:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    fixed_batch = sess.get_inputs()[0].shape[0] == 1

    def fn(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if fixed_batch:
            return np.concatenate(
                [
                    sess.run(None, {"obs_hist": obs[i : i + 1], "goal": goal[i : i + 1]})[0]
                    for i in range(obs.shape[0])
                ]
            )
        return sess.run(None, {"obs_hist": obs, "goal": goal})[0]

    return fn


def run_benchmark(
    make_env: Callable[[int, dict[str, Any]], Any],
    motion_files: Sequence[str],
    policy_fn: PolicyFn,
    num_envs: int = 256,
    verbose: bool = True,
) -> list[ClipResult]:
    """Play every clip of ``motion_files`` once; ``make_env(num_envs, overrides)`` builds an ``H2SuperMimic`` env."""
    num_envs = min(num_envs, len(motion_files))
    env = make_env(
        num_envs, {**EVAL_ENV_OVERRIDES, "motion_file": list(motion_files), "motion_dirs": []}
    )
    lib = env.library
    n_clips = lib.num_clips
    max_r = env.cfg.reward.max_weight_sum * env.cfg.ctrl_dt
    current = np.arange(num_envs)
    nxt = current + num_envs
    env.forced_frame[:] = lib.clip_start[current]
    state = env.init_state()
    env.forced_frame[:] = lib.clip_start[np.minimum(nxt, n_clips - 1)]
    active = np.ones(num_envs, dtype=bool)
    ret_sum = np.zeros(num_envs)
    steps = np.zeros(num_envs, dtype=np.int64)
    term_names: list[str] = []
    term_sum = None
    results: dict[int, ClipResult] = {}
    while len(results) < n_clips:
        state = env.step(policy_fn(state.obs["obs"], state.obs["goal"]).astype(np.float32))
        if term_sum is None:
            term_names = list(env.sm_reward_terms)
            term_sum = np.zeros((num_envs, len(term_names)))
        ret_sum += state.reward
        term_sum += np.stack([env.sm_reward_terms[k] for k in term_names], axis=-1)
        steps += 1
        done = (state.terminated | state.truncated) & active
        for e in np.flatnonzero(done):
            ci = int(current[e])
            if ci < n_clips and ci not in results:
                st = int(steps[e])
                results[ci] = ClipResult(
                    name=lib.clip_names[ci],
                    frames=int(lib.clip_len[ci]),
                    steps=st,
                    norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                    failed=bool(state.terminated[e]),
                    terms={k: float(term_sum[e, i]) / max(st, 1) for i, k in enumerate(term_names)},
                )
                if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                    print(f"  evaluated {len(results)}/{n_clips}", flush=True)
            # The auto-reset already consumed forced_frame: queue the following clip.
            current[e] = nxt[e]
            nxt[e] += num_envs
            active[e] = current[e] < n_clips
            env.forced_frame[e] = lib.clip_start[min(nxt[e], n_clips - 1)]
            ret_sum[e] = 0.0
            term_sum[e] = 0.0
            steps[e] = 0
    env.close()
    return [results[i] for i in range(n_clips)]


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    tot_steps = sum(r.steps for r in results)
    out = {
        "clips": n,
        "failed": sum(r.failed for r in results),
        "mean_norm_return": sum(r.norm_return for r in results) / max(n, 1),
        "step_weighted_norm_return": sum(r.norm_return * r.steps for r in results)
        / max(tot_steps, 1),
    }
    if results:
        out["terms"] = {
            k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1)
            for k in results[0].terms
        }
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1)
    )
