"""Export the policy of a SuperMimic training checkpoint to ONNX (+ json sidecar).

Usage:
    uv run scripts/supermimic/export_onnx.py --checkpoint <run>/ckpt_latest.pt [--out <run>/policy.onnx] [--dynamic]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np

from unilab.algos.torch.supermimic.benchmark import load_policy_from_checkpoint
from unilab.algos.torch.supermimic.export import export_policy_onnx


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--dynamic", action="store_true", help="dynamic batch dimension (default: batch 1)"
    )
    args = p.parse_args()
    policy = load_policy_from_checkpoint(args.checkpoint, "cpu")
    # Deployment contract (joint order, action scale, default pose) comes from the env owner.
    import mujoco

    from unilab.envs.motion_tracking.h2.supermimic import H2SuperMimicCfg, compute_action_scale

    cfg = H2SuperMimicCfg()
    model = mujoco.MjModel.from_xml_path(cfg.scene.model_file)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    default_pos = model.key_qpos[key, 7:].tolist()
    scale = compute_action_scale(model, cfg.action_scale_factor).tolist()
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".onnx")
    export_policy_onnx(
        policy,
        out,
        {
            "robot": "h2",
            "joint_names": list(cfg.joint_names),
            "action_scale": scale,
            "default_joint_pos": default_pos,
            "obs_terms": ["joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "actions"],
        },
        dynamic_batch=args.dynamic,
    )
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(str(out))
    obs = np.random.randn(1, policy.cfg.obs_dim).astype(np.float32)
    goal = np.random.randn(1, policy.cfg.goal_dim).astype(np.float32)
    ref = policy(torch.from_numpy(obs), torch.from_numpy(goal)).detach().numpy()
    err = float(np.abs(sess.run(None, {"obs_hist": obs, "goal": goal})[0] - ref).max())
    print(f"[export] wrote {out} (+ .json), onnxruntime max |diff| = {err:.2e}")
    assert err < 1e-4


if __name__ == "__main__":
    main()
