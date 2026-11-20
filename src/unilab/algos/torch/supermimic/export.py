"""ONNX export of a SuperMimic policy: ``action = policy(obs_hist, goal)``.

Inputs ``obs_hist [B, H*(3N+6)]`` (term-major history) and ``goal [B, K*(2N+10)]``; output
``action [B, N]``. Input scaling is inside the graph; the joint target is
``default_joint_pos + action * action_scale`` (listed in the json sidecar).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .nets import Policy


def export_policy_onnx(
    policy: Policy, path: str | Path, metadata: dict[str, Any], dynamic_batch: bool = False
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = policy.eval().cpu()
    obs = torch.zeros(1, policy.cfg.obs_dim)
    goal = torch.zeros(1, policy.cfg.goal_dim)
    axes = (
        {"obs_hist": {0: "batch"}, "goal": {0: "batch"}, "action": {0: "batch"}}
        if dynamic_batch
        else None
    )
    torch.onnx.export(
        policy,
        (obs, goal),
        str(path),
        input_names=["obs_hist", "goal"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes=axes,
        dynamo=False,
    )
    sidecar = {
        "obs_dim": policy.cfg.obs_dim,
        "goal_dim": policy.cfg.goal_dim,
        "num_joints": policy.cfg.num_joints,
        **metadata,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return path
