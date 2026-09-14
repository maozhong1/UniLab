"""Standalone verification for the byte-exact official SONIC critic warm-load.

Checks (no sim / no XPU needed — runs on CPU in seconds):

  1. DIM ARITHMETIC — the official privileged_mf_hist critic obs sums to 1645, and the
     env's ``_critic_mf_hist_dim`` formula reproduces it (n=29, F=10, H=10, 14 bodies).
  2. CHECKPOINT — last.pt's ``value_state_dict`` critic first layer is (2048, 1645), i.e.
     the official 6-hidden-layer [2048,2048,1024,1024,512,512] MLP over a 1645-d obs.
  3. LOAD — ``SonicCriticModel(pretrained_ckpt=last.pt)`` on a 1645-d critic obs loads the
     mlp weights EXACTLY (allclose) + copies the RunningMeanStd stats, and forward() is
     finite with shape (B, 1).
  4. GUARD — loading into a wrong-width (926) critic raises a clear ValueError.

Run:
    HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python scripts/sonic/critic_load_check.py \
        --ckpt /home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/sonic_release/last.pt
"""
from __future__ import annotations

import argparse

import torch
from tensordict import TensorDict

from unilab.algos.torch.sonic import SonicCriticModel
from unilab.algos.torch.sonic.core import _install_fake_import_hook

CRITIC_OBS_DIM = 1645


def _official_dim(n: int = 29, F: int = 10, H: int = 10, nb: int = 14) -> int:
    # command_multi_future(2*F*n) + anchor_pos(3) + anchor_ori(6) + body(9*nb) + hist((6+3n)*H)
    return 2 * F * n + 3 + 6 + 9 * nb + (6 + 3 * n) * H


def _make_critic(ckpt: str | None, obs_dim: int) -> SonicCriticModel:
    obs = TensorDict({"critic": torch.zeros(2, obs_dim)}, batch_size=[2])
    return SonicCriticModel(
        obs,
        {"critic": ["critic"]},
        "critic",
        1,
        hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        activation="swish",
        obs_normalization=True,
        pretrained_ckpt=ckpt,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/sonic_release/last.pt",
    )
    args = ap.parse_args()

    # 1. dim arithmetic
    assert _official_dim() == CRITIC_OBS_DIM, _official_dim()
    print(f"[1] dim arithmetic OK: official privileged_mf_hist critic obs = {CRITIC_OBS_DIM}")

    # 2. checkpoint shapes
    _install_fake_import_hook()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vsd = ck["value_state_dict"]
    w0 = vsd["critic_module.module.0.weight"]
    assert tuple(w0.shape) == (2048, CRITIC_OBS_DIM), tuple(w0.shape)
    print(f"[2] checkpoint OK: value_state_dict critic_module.module.0.weight = {tuple(w0.shape)}")

    # 3. load into a 1645-d critic + verify exactness and finite forward
    critic = _make_critic(args.ckpt, CRITIC_OBS_DIM)
    assert torch.allclose(critic.mlp[0].weight.detach().cpu(), w0), "mlp.0.weight != ckpt"
    assert torch.allclose(critic.mlp[12].weight.detach().cpu(), vsd["critic_module.module.12.weight"])
    mean = vsd["running_mean_std.running_mean"].reshape(1, -1)
    assert torch.allclose(critic.obs_normalizer._mean.detach().cpu(), mean), "normalizer mean not copied"
    x = TensorDict({"critic": torch.randn(4, CRITIC_OBS_DIM)}, batch_size=[4])
    out = critic(x)
    assert out.shape == (4, 1) and torch.isfinite(out).all(), out.shape
    print(f"[3] load OK: mlp weights match ckpt exactly, RMS stats copied, forward -> {tuple(out.shape)} finite")

    # 4. dim guard fires on wrong width
    try:
        _make_critic(args.ckpt, 926)
    except ValueError as e:
        print(f"[4] guard OK: wrong-width (926) critic rejected -> {str(e)[:70]}...")
    else:
        raise AssertionError("expected ValueError for 926-d critic load")

    print("\nALL CHECKS PASSED — last.pt critic warm-loads into the 1645-d SonicCriticModel.")


if __name__ == "__main__":
    main()
