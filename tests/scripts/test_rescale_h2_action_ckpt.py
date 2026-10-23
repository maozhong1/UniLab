from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from unilab.algos.torch.sonic.core import SonicG1Core

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sonic" / "rescale_h2_action_ckpt.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rescale_h2_action_ckpt", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _core() -> SonicG1Core:
    return SonicG1Core(with_kin_aux=False, use_fsq=True, enc_input_dim=640, proprio_dim=930,
                       action_dim=29).eval()


def _write_ckpt(path: Path, core: SonicG1Core) -> None:
    actor = {f"core.{k}": v for k, v in core.state_dict().items()}
    actor["distribution.std_param"] = torch.full((29,), 0.3)
    params = [v for k, v in actor.items() if "fsq" not in k]
    opt = {"state": {i: {"step": torch.tensor(1.0), "exp_avg": torch.ones_like(p),
                         "exp_avg_sq": torch.ones_like(p)} for i, p in enumerate(params)},
           "param_groups": [{"params": list(range(len(params)))}]}
    critic = {"obs_normalizer._mean": torch.randn(1, 1645), "obs_normalizer._var": torch.rand(1, 1645) + 0.5}
    critic["obs_normalizer._std"] = critic["obs_normalizer._var"].sqrt()
    torch.save({"actor_state_dict": actor, "critic_state_dict": critic, "optimizer_state_dict": opt,
                "iter": 6000}, path)


def test_rescaled_checkpoint_commands_identical_pd_targets(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    torch.manual_seed(0)
    src_run = tmp_path / "run"
    src_run.mkdir()
    old_core = _core()
    _write_ckpt(src_run / "model_6000.pt", old_core)

    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src_run / "model_6000.pt"), "--dst-run", "out"])
    mod.main()
    out = torch.load(tmp_path / "out" / "model_6000.pt", map_location="cpu", weights_only=False)

    new_core = _core()
    new_core.load_state_dict({k[5:]: v for k, v in out["actor_state_dict"].items() if k.startswith("core.")})
    f = torch.as_tensor(mod.OLD_ACTION_SCALE / np.asarray(mod._SONIC_ACTION_SCALE_MUJOCO), dtype=torch.float32)
    assert (f != 1).sum() == 8  # ankles (4) + wrist pitch/yaw (4)

    # The new policy sees the action history in new units (old action * f) and must
    # output old action * f, i.e. command the identical PD target.
    enc, proprio = torch.randn(64, 640), torch.randn(64, 930)
    proprio_new = proprio.clone()
    a0 = mod._DEC_ACT0 - mod._TOKEN
    proprio_new[:, a0:a0 + 29 * mod._H] *= f.repeat(mod._H)
    with torch.no_grad():
        target_old = old_core.act_mean(enc, proprio) * torch.as_tensor(mod.OLD_ACTION_SCALE, dtype=torch.float32)
        target_new = new_core.act_mean(enc, proprio_new) * torch.as_tensor(
            np.asarray(mod._SONIC_ACTION_SCALE_MUJOCO), dtype=torch.float32)
    torch.testing.assert_close(target_new, target_old, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out["actor_state_dict"]["distribution.std_param"], 0.3 * f)

    # Critic normalizer: last_actions history scaled by f, everything else untouched.
    src = torch.load(src_run / "model_6000.pt", map_location="cpu", weights_only=False)["critic_state_dict"]
    gain = torch.ones(1645)
    gain[-29 * mod._H:] = f.repeat(mod._H)
    torch.testing.assert_close(out["critic_state_dict"]["obs_normalizer._mean"], src["obs_normalizer._mean"] * gain)
    torch.testing.assert_close(out["critic_state_dict"]["obs_normalizer._std"], src["obs_normalizer._std"] * gain)
