"""
SONIC G1-only core network for V2 finetune on UniLab (Intel GPU).

Faithful reproduction of the **G1 path** of sonic's universal-token policy:
    G1 encoder -> FSQ(2x32, 32 levels) -> g1_dyn decoder.
Can cleanly load the pretrained weights from the official ``sonic_release/last.pt``
(verified strict=OK for encoder / g1_dyn / g1_kin / std).

Architecture (from last.pt + config.yaml):
  encoders.g1.module : Linear[640->2048->1024->512->512->64], SiLU, no obs-norm
  FSQ                : vector_quantize_pytorch.FSQ, levels=[32]*32, 2 tokens -> 64
  decoders.g1_dyn    : Linear[994->2048->2048->1024->1024->512->512->29]  (994 = token64 + proprio930)
  decoders.g1_kin    : Linear[64->2048->1024->512->512->640]  (g1_recon aux, optional)
  std                : [29], init 0.05, clamp [0.001, 0.5]

This is the *core network only*. The UniLab RSL-RL ActorCritic contract wrapper
(act/forward/distribution/obs-groups/as_onnx ...) lives in ``models.py``.

``use_fsq=False`` bypasses quantization (continuous token = raw encoder output) for
the task-5 no-FSQ baseline. Note: FSQ tokens are normalized to ~[-1, 1]; the raw
encoder output is unbounded, so a decoder trained with FSQ off is NOT interchangeable
with one trained with FSQ on -- use the flag only for from-scratch baselines.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# FSQ -- faithful reimpl of vector_quantize_pytorch.FSQ (offset=0.5 for even levels)
# If vector_quantize_pytorch is installed we prefer the official impl so the token
# space matches the deployed encoder exactly.
# ----------------------------------------------------------------------------
def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradient."""
    return z + (z.round() - z).detach()


class FSQFallback(nn.Module):
    """Self-contained FSQ, algorithm-aligned with vector_quantize_pytorch. No params.

    ``half_l``/``offset``/``shift``/``half_width`` depend only on ``levels`` (fixed),
    so they are precomputed as buffers at init. This keeps ``forward`` free of
    ``atanh`` (unsupported by ONNX opset<=18) -- export-clean and numerically identical.
    """

    def __init__(self, levels: list[int], eps: float = 1e-3) -> None:
        super().__init__()
        lv = torch.tensor(levels, dtype=torch.float32)
        half_l = (lv - 1) * (1 + eps) / 2
        offset = torch.where(lv % 2 == 0, torch.full_like(lv, 0.5), torch.zeros_like(lv))
        shift = torch.atanh(offset / half_l)  # constant, computed once at init
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("_half_l", half_l)
        self.register_buffer("_offset", offset)
        self.register_buffer("_shift", shift)
        self.register_buffer("_half_width", (torch.tensor(levels, dtype=torch.long) // 2).float())
        self.codebook_dim = len(levels)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        return (z + self._shift).tanh() * self._half_l - self._offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        q = round_ste(self.bound(z))
        return q / self._half_width  # normalize to ~[-1, 1]


def make_fsq(levels: list[int]) -> tuple[nn.Module, bool]:
    try:
        from vector_quantize_pytorch import FSQ as _FSQ  # sonic's original lib

        return _FSQ(levels=list(levels)), True
    except Exception:
        return FSQFallback(list(levels)), False


# ----------------------------------------------------------------------------
def build_mlp(dims: list[int], act: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class SonicG1Core(nn.Module):
    """G1 encoder + FSQ + g1_dyn decoder (+ optional g1_kin aux head)."""

    G1_ENC_DIMS = [640, 2048, 1024, 512, 512, 64]
    G1_DYN_DIMS = [994, 2048, 2048, 1024, 1024, 512, 512, 29]
    G1_KIN_DIMS = [64, 2048, 1024, 512, 512, 640]
    NUM_TOKENS = 2
    FSQ_DIM_PER_TOKEN = 32
    FSQ_LEVELS = 32
    TOKEN_DIM = NUM_TOKENS * FSQ_DIM_PER_TOKEN  # 64
    ENC_INPUT_DIM = G1_ENC_DIMS[0]  # 640
    PROPRIO_DIM = 930
    ACTION_DIM = 29

    def __init__(self, with_kin_aux: bool = False, use_fsq: bool = True) -> None:
        super().__init__()
        self.use_fsq = bool(use_fsq)
        self.encoder = build_mlp(self.G1_ENC_DIMS)  # 640 -> 64
        self.fsq, self.fsq_is_official = make_fsq([self.FSQ_LEVELS] * self.FSQ_DIM_PER_TOKEN)
        self.decoder = build_mlp(self.G1_DYN_DIMS)  # 994 -> 29
        self.kin = build_mlp(self.G1_KIN_DIMS) if with_kin_aux else None  # 64 -> 640 (aux)
        # informational: sonic's own action std (used only if you want to seed the
        # RSL-RL GaussianDistribution). The RL policy std is owned by the distribution.
        self.log_std = nn.Parameter(torch.full((self.ACTION_DIM,), 0.05).log())

    def encode(self, obs_g1: torch.Tensor) -> torch.Tensor:
        """(B, 640) -> token (B, 64). FSQ-quantized unless ``use_fsq`` is False."""
        z = self.encoder(obs_g1)  # (B, 64) continuous
        if not self.use_fsq:
            return z
        z = z.view(-1, self.NUM_TOKENS, self.FSQ_DIM_PER_TOKEN)  # (B, 2, 32)
        tok = self.fsq(z)  # (B, 2, 32) quantized
        return tok.reshape(-1, self.TOKEN_DIM)  # (B, 64)

    def decode(self, token: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """(B, 64) token ++ (B, 930) proprio -> action mean (B, 29)."""
        return self.decoder(torch.cat([token, proprio], dim=-1))

    def act_mean(self, obs_g1: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Full G1 path: encoder -> FSQ -> decoder -> action mean (B, 29)."""
        return self.decode(self.encode(obs_g1), proprio)

    def recon(self, token: torch.Tensor) -> torch.Tensor:
        """g1_kin aux: token (B, 64) -> reconstructed encoder input (B, 640)."""
        assert self.kin is not None, "g1_kin aux head not built (with_kin_aux=False)"
        return self.kin(token)

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp().clamp(0.001, 0.5)


# ----------------------------------------------------------------------------
# Load the G1 weight subset from sonic_release/last.pt (policy_state_dict).
# The checkpoint pickles reference sonic/isaaclab modules; a fake-import hook lets
# torch.load unpickle it without those packages installed.
# ----------------------------------------------------------------------------
def _install_fake_import_hook() -> None:
    class _Any:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            if isinstance(s, dict):
                self.__dict__.update(s)

        def __setitem__(self, k, v):
            self.__dict__[k] = v

    class FM(types.ModuleType):
        def __init__(self, n):
            super().__init__(n)
            self.__file__ = "<f:%s>" % n
            self.__path__ = []
            self.__all__ = []

        def __getattr__(self, n):
            if n.startswith("__") and n.endswith("__"):
                raise AttributeError(n)
            c = type(n, (_Any,), {})
            setattr(self, n, c)
            return c

    FAKE = (
        "trl", "gear_sonic", "groot", "isaaclab", "isaacsim", "omni", "pxr", "warp",
        "smpl_sim",
        # HF stack faked too: the UniLab venv lacks these; we only read the tensor
        # policy_state_dict, so stub classes are safe.
        "transformers", "accelerate", "peft", "safetensors", "huggingface_hub",
        "tokenizers", "datasets", "deepspeed", "bitsandbytes", "sentencepiece",
    )

    class L(importlib.abc.Loader):
        def create_module(self, s):
            return FM(s.name)

        def exec_module(self, m):
            pass

    class F(importlib.abc.MetaPathFinder):
        def find_spec(self, fn, p, t=None):
            if fn.split(".")[0] in FAKE:
                return importlib.machinery.ModuleSpec(fn, L(), is_package=True)
            return None

    try:
        import transformers  # noqa: F401  (pre-import real lib to avoid unpickle traps)
    except Exception:
        pass
    sys.meta_path.insert(0, F())


def load_g1_from_last_pt(model: SonicG1Core, ckpt_path: str) -> SonicG1Core:
    """Load encoder / g1_dyn decoder (+ optional g1_kin + std) from last.pt (strict)."""
    _install_fake_import_hook()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    psd = ck["policy_state_dict"]

    def sub(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in psd.items() if k.startswith(prefix)}

    model.encoder.load_state_dict(sub("actor_module.encoders.g1.module."), strict=True)
    model.decoder.load_state_dict(sub("actor_module.decoders.g1_dyn.module."), strict=True)
    if model.kin is not None:
        model.kin.load_state_dict(sub("actor_module.decoders.g1_kin.module."), strict=True)
    if "std" in psd and psd["std"].shape == model.log_std.shape:
        with torch.no_grad():
            model.log_std.copy_(psd["std"].clamp_min(1e-4).log())
    return model


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        # Default to ./last.pt (repo root when launched via `uv run python scripts/...`).
        # Override with --ckpt to point at any sonic_release/last.pt copy.
        default="./last.pt",
    )
    ap.add_argument("--aux", action="store_true", help="build g1_kin recon head")
    ap.add_argument("--no-fsq", action="store_true", help="bypass FSQ (continuous token)")
    a = ap.parse_args()
    m = SonicG1Core(with_kin_aux=a.aux, use_fsq=not a.no_fsq).eval()
    load_g1_from_last_pt(m, a.ckpt)
    print(f"FSQ backend: {'vector_quantize_pytorch (official)' if m.fsq_is_official else 'FSQFallback (faithful reimpl)'}")
    print(f"use_fsq: {m.use_fsq}")
    with torch.no_grad():
        obs_g1 = torch.randn(4, 640)
        proprio = torch.randn(4, 930)
        tok = m.encode(obs_g1)
        act = m.act_mean(obs_g1, proprio)
        uniq = torch.unique(tok).numel()
        print("token:", tuple(tok.shape), "| action:", tuple(act.shape), "| std[:3]:", m.std[:3].tolist())
        print("token distinct values (<=32 => quantized):", uniq)
        if a.aux:
            print("g1_kin recon:", tuple(m.recon(tok).shape))
    print("OK: SonicG1Core loads last.pt (strict) + forward works.")
