"""Task-3 verification: build G1SonicMotionTracking, reset + step, check sonic obs dims.

Exercises both obs paths: init_state (reset -> history fill) and step (history push).
Run:  uv run --no-sync python scripts/sonic/env_smoke.py
"""
from __future__ import annotations

import numpy as np

from unilab.base import registry

registry.ensure_registries()  # discover + import all registry modules (as training does)

ENC, PROPRIO, CRITIC = 640, 930, 286
ACTOR = ENC + PROPRIO  # 1570


def main() -> None:
    N = 4
    env = registry.make("G1SonicMotionTracking", num_envs=N, sim_backend="mujoco")
    try:
        spec = env.obs_groups_spec
        print("obs_groups_spec:", spec)
        assert spec["obs"] == ACTOR, spec
        assert spec["critic"] == CRITIC, spec
        assert env.observation_space.shape[0] == ACTOR + CRITIC

        state = env.init_state()  # reset-all -> history fill path
        assert state.obs["obs"].shape == (N, ACTOR), state.obs["obs"].shape
        assert state.obs["critic"].shape == (N, CRITIC), state.obs["critic"].shape
        assert np.isfinite(state.obs["obs"]).all(), "non-finite in reset obs"
        print("reset obs OK:", {k: v.shape for k, v in state.obs.items()})

        act_dim = env.action_space.shape[0]
        assert act_dim == 29, act_dim
        # a few steps -> history push path; obs should stay finite and shaped
        for i in range(5):
            state = env.step(np.zeros((N, act_dim), dtype=np.float32))
            assert state.obs["obs"].shape == (N, ACTOR)
            assert state.obs["critic"].shape == (N, CRITIC)
            assert np.isfinite(state.obs["obs"]).all(), f"non-finite at step {i}"
        print("step obs OK after 5 steps")

        # history sanity: after >1 distinct steps the proprio history block should
        # NOT be all-identical across frames (i.e., the ring actually advanced).
        proprio = state.obs["obs"][:, ENC:]  # (N, 930)
        gyro_hist = proprio[:, :30].reshape(N, 10, 3)  # his_base_angular_velocity
        spread = float(np.abs(gyro_hist[:, -1] - gyro_hist[:, 0]).max())
        print(f"gyro history frame0-vs-frame9 max|Δ| = {spread:.4e} (>0 => ring advanced)")

        print("\nTASK-3 ENV SMOKE PASSED")
    finally:
        env.close()


if __name__ == "__main__":
    main()
