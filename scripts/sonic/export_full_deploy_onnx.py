"""Deploy-split ONNX export for a FROM-SCRATCH (MuJoCo-joint-order) SONIC G1 policy.

Use this INSTEAD of ``export_deploy_onnx.py`` when the checkpoint was trained with
``mujoco_to_isaaclab_perm=false`` AND ``action_output_isaaclab_to_mujoco=false``
(i.e. the net consumes/produces joints in MuJoCo order — the full_train from-scratch
default). The sonic deploy runtime is hard-wired to IsaacLab joint order, so a plain
export of such a model drives the WRONG joints and the robot cannot stand.

This script bakes the MuJoCo<->IsaacLab joint permutation INTO the exported graphs so the
deploy's IsaacLab-order I/O is bridged with no C++/config change and no retrain:

  * encoder (1762 & 640): the deploy feeds command joints in IsaacLab order -> we permute
    each 29-joint block IsaacLab->MuJoCo BEFORE the (legacy) reshape into the encoder.
  * decoder (994): the deploy feeds proprio jpos/jvel/last_action in IsaacLab order -> we
    permute those blocks IsaacLab->MuJoCo before decode; the net's MuJoCo-order action
    output is permuted MuJoCo->IsaacLab on the way out, so the deploy's own
    isaaclab->mujoco action remap lands on the correct actuators.

Non-joint terms (anchor 6D, base gyro, gravity dir) are NOT permuted.

Permutation is a lossless reindex, so this is mathematically equivalent to having trained
with perms=true — it just avoids a retrain. (For NEW deploy-intended runs, prefer training
with both perms=true and using export_deploy_onnx.py.)

Run:
    uv run --no-sync python scripts/sonic/export_full_deploy_onnx.py \
        --ckpt ./logs/.../model_4600.pt --merged-1762 --out-dir ./model_onnx2/ --verify
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from unilab.algos.torch.sonic.core import SonicG1Core

# Reuse the deploy buffer offsets + ORT helpers + loader from the plain exporter.
from export_deploy_onnx import (  # noqa: E402
    OPSET,
    TOL,
    MERGED_INPUT_DIM,
    _report,
    load_core,
)
from parity_harness import (  # noqa: E402
    ANCHOR_PER_FRAME,
    CMD_PER_FRAME,
    N_FUT,
    OFF_ANCHOR,
    OFF_CMD,
    OFF_MODE,
    ENC_TOTAL,
    _ort_run,
)

# ---------------------------------------------------------------------------
# Joint permutation (single source of truth = tracking_sonic; inline fallback).
#   _MUJOCO_TO_ISAACLAB: v_isaaclab = v_mujoco[_MUJOCO_TO_ISAACLAB]
#   _ISAACLAB_TO_MUJOCO: v_mujoco   = v_isaaclab[_ISAACLAB_TO_MUJOCO]  (inverse)
# ---------------------------------------------------------------------------
try:
    from unilab.envs.motion_tracking.g1.tracking_sonic import (  # noqa: E402
        _ISAACLAB_TO_MUJOCO as _I2M_NP,
        _MUJOCO_TO_ISAACLAB as _M2I_NP,
    )
except Exception:  # pragma: no cover - fallback keeps the exporter standalone
    _M2I_NP = np.array(
        [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
         16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
        dtype=np.intp,
    )
    _I2M_NP = np.empty(29, dtype=np.intp)
    _I2M_NP[_M2I_NP] = np.arange(29, dtype=np.intp)

# sanity: the two perms MUST be exact inverses, else the bridge is wrong.
assert np.array_equal(np.asarray(_I2M_NP)[np.asarray(_M2I_NP)], np.arange(29)), "perm not inverse"

IDX_IN = torch.as_tensor(np.asarray(_I2M_NP), dtype=torch.long)   # IsaacLab -> MuJoCo (inputs)
IDX_OUT = torch.as_tensor(np.asarray(_M2I_NP), dtype=torch.long)  # MuJoCo -> IsaacLab (action out)

NACT = 29
H = SonicG1Core.PROPRIO_DIM // (3 + NACT + NACT + NACT + 3)  # =10 history frames
assert (3 + NACT + NACT + NACT + 3) * H == SonicG1Core.PROPRIO_DIM, SonicG1Core.PROPRIO_DIM

# proprio(930) field offsets: [gyro(3) | jpos(29) | jvel(29) | last_action(29) | gravity(3)] x H
_G = 3 * H                       # gyro block end          (0:30)
_JP = _G + NACT * H              # jpos block end          (30:320)
_JV = _JP + NACT * H             # jvel block end          (320:610)
_LA = _JV + NACT * H             # last_action block end   (610:900)
# gravity is _LA:930


def _perm_blocks(x: torch.Tensor, nblocks: int, idx: torch.Tensor) -> torch.Tensor:
    """Permute the 29-joint dim within each of ``nblocks`` contiguous joint blocks.

    x: (B, nblocks*29) -> reshape (B, nblocks, 29) -> gather idx on last dim -> flatten.
    """
    B = x.shape[0]
    return x.reshape(B, nblocks, NACT)[:, :, idx].reshape(B, nblocks * NACT)


# ---------------------------------------------------------------------------
# Export wrappers with the joint permutation baked in.
# ---------------------------------------------------------------------------
class MergedEncoderExportPerm(nn.Module):
    """Deploy drop-in encoder WITH joint bridge: obs_dict[B,1762] -> encoded_tokens[B,64].

    Same 1762->640 transform as export_deploy_onnx.MergedEncoderExport, but the command
    block (580 = 20 x 29 joint vectors) is permuted IsaacLab->MuJoCo before the legacy
    (10,58) reshape, so the MuJoCo-trained encoder sees the joint order it expects.
    """

    input_names = ["obs_dict"]
    output_names = ["encoded_tokens"]

    def __init__(self, core: SonicG1Core) -> None:
        super().__init__()
        self.core = core

    def forward(self, obs_dict: torch.Tensor) -> torch.Tensor:
        B = obs_dict.shape[0]
        nf, cpf, af = N_FUT, CMD_PER_FRAME, ANCHOR_PER_FRAME  # 10, 58, 6
        cmd580 = obs_dict[:, OFF_CMD : OFF_CMD + nf * cpf]            # (B,580) IsaacLab
        cmd580 = _perm_blocks(cmd580, 2 * nf, IDX_IN)                # -> MuJoCo (20 blocks)
        cmd = cmd580.reshape(B, nf, cpf)                            # legacy (10,58) packing
        anch = obs_dict[:, OFF_ANCHOR : OFF_ANCHOR + nf * af].reshape(B, nf, af)
        enc640 = torch.cat([cmd, anch], dim=-1).reshape(B, nf * (cpf + af))
        return self.core.encode(enc640)


class EncoderExportPerm(nn.Module):
    """Split g1 encoder WITH joint bridge: obs_dict[B,640] -> encoded_tokens[B,64].

    The 640 = 10 x [cmd58, anchor6]; each cmd58 = two 29-joint vectors. Permute the joint
    halves IsaacLab->MuJoCo, leave the 6-d anchor untouched.
    """

    input_names = ["obs_dict"]
    output_names = ["encoded_tokens"]

    def __init__(self, core: SonicG1Core) -> None:
        super().__init__()
        self.core = core

    def forward(self, obs_dict: torch.Tensor) -> torch.Tensor:
        B = obs_dict.shape[0]
        nf, cpf, af = N_FUT, CMD_PER_FRAME, ANCHOR_PER_FRAME
        x = obs_dict.reshape(B, nf, cpf + af)              # (B,10,64)
        cmd = x[:, :, :cpf].reshape(B, nf * 2, NACT)[:, :, IDX_IN].reshape(B, nf, cpf)
        anch = x[:, :, cpf:]                               # (B,10,6) untouched
        enc640 = torch.cat([cmd, anch], dim=-1).reshape(B, nf * (cpf + af))
        return self.core.encode(enc640)


class DecoderExportPerm(nn.Module):
    """Decoder WITH joint bridge: obs_dict[B,994]=token[64]++proprio[930] -> action[B,29].

    proprio jpos/jvel/last_action (each 10x29) permuted IsaacLab->MuJoCo before decode;
    the MuJoCo-order action output is permuted MuJoCo->IsaacLab so the deploy's own
    isaaclab->mujoco action remap lands on the correct actuators.
    """

    input_names = ["obs_dict"]
    output_names = ["action"]

    def __init__(self, core: SonicG1Core) -> None:
        super().__init__()
        self.core = core

    def forward(self, obs_dict: torch.Tensor) -> torch.Tensor:
        token = obs_dict[:, : SonicG1Core.TOKEN_DIM]
        proprio = obs_dict[:, SonicG1Core.TOKEN_DIM :]     # (B,930) IsaacLab joints
        gyro = proprio[:, :_G]
        jpos = _perm_blocks(proprio[:, _G:_JP], H, IDX_IN)
        jvel = _perm_blocks(proprio[:, _JP:_JV], H, IDX_IN)
        lastact = _perm_blocks(proprio[:, _JV:_LA], H, IDX_IN)
        grav = proprio[:, _LA:]
        proprio_m = torch.cat([gyro, jpos, jvel, lastact, grav], dim=1)  # -> MuJoCo
        action_m = self.core.decode(token, proprio_m)                    # MuJoCo-order action
        return action_m[:, IDX_OUT]                                      # -> IsaacLab out


# ---------------------------------------------------------------------------
# Export.
# ---------------------------------------------------------------------------
def export_split(core: SonicG1Core, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    enc_path = os.path.join(out_dir, "model_encoder_g1.onnx")
    dec_path = os.path.join(out_dir, "model_decoder.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            EncoderExportPerm(core).eval(),
            (torch.zeros(1, SonicG1Core.ENC_INPUT_DIM),),
            enc_path,
            input_names=EncoderExportPerm.input_names,
            output_names=EncoderExportPerm.output_names,
            opset_version=OPSET,
        )
        torch.onnx.export(
            DecoderExportPerm(core).eval(),
            (torch.zeros(1, SonicG1Core.TOKEN_DIM + SonicG1Core.PROPRIO_DIM),),
            dec_path,
            input_names=DecoderExportPerm.input_names,
            output_names=DecoderExportPerm.output_names,
            opset_version=OPSET,
        )
    print(f"Exported (perm) encoder ONNX -> {enc_path}")
    print(f"Exported (perm) decoder ONNX -> {dec_path}")
    return enc_path, dec_path


def export_merged_encoder(core: SonicG1Core, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model_encoder.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            MergedEncoderExportPerm(core).eval(),
            (torch.zeros(1, MERGED_INPUT_DIM),),
            path,
            input_names=MergedEncoderExportPerm.input_names,
            output_names=MergedEncoderExportPerm.output_names,
            opset_version=OPSET,
        )
    print(f"Exported (perm) merged-1762 encoder ONNX -> {path}")
    return path


# ---------------------------------------------------------------------------
# Verify: exported ONNX reproduces "permute-then-core" exactly (export integrity).
# ---------------------------------------------------------------------------
def verify_self(core: SonicG1Core, enc_g1_path: str, dec_path: str, merged_path: str | None) -> bool:
    print("Self-verify (exported perm-ONNX vs torch permute+core)")
    rng = np.random.default_rng(0)
    B = 8
    ok = True

    # decoder: random 994 (proprio in IsaacLab order) -> onnx vs permute+decode+permute
    token = rng.uniform(-1, 1, (B, SonicG1Core.TOKEN_DIM)).astype(np.float32)
    proprio = rng.standard_normal((B, SonicG1Core.PROPRIO_DIM)).astype(np.float32)
    obs994 = np.concatenate([token, proprio], axis=1)
    onnx_a = _ort_run(ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"]), obs994)
    with torch.no_grad():
        ref_a = DecoderExportPerm(core).eval()(torch.from_numpy(obs994)).numpy()
    ok &= _report("decoder(994) perm-onnx vs ref", ref_a, onnx_a)

    # split g1 encoder: random 640 -> onnx vs permute+encode
    enc640 = rng.standard_normal((B, SonicG1Core.ENC_INPUT_DIM)).astype(np.float32)
    onnx_t = _ort_run(ort.InferenceSession(enc_g1_path, providers=["CPUExecutionProvider"]), enc640)
    with torch.no_grad():
        ref_t = EncoderExportPerm(core).eval()(torch.from_numpy(enc640)).numpy()
    ok &= _report("encoder_g1(640) perm-onnx vs ref", ref_t, onnx_t)

    if merged_path:
        buf = np.zeros((B, ENC_TOTAL), dtype=np.float32)
        buf[:, OFF_MODE] = 0.0
        buf[:, OFF_CMD : OFF_CMD + N_FUT * CMD_PER_FRAME] = rng.standard_normal(
            (B, N_FUT * CMD_PER_FRAME)).astype(np.float32)
        buf[:, OFF_ANCHOR : OFF_ANCHOR + N_FUT * ANCHOR_PER_FRAME] = rng.standard_normal(
            (B, N_FUT * ANCHOR_PER_FRAME)).astype(np.float32)
        onnx_m = _ort_run(ort.InferenceSession(merged_path, providers=["CPUExecutionProvider"]), buf)
        with torch.no_grad():
            ref_m = MergedEncoderExportPerm(core).eval()(torch.from_numpy(buf)).numpy()
        ok &= _report("merged(1762) perm-onnx vs ref", ref_m, onnx_m)
    return ok


def verify_perm_roundtrip(core: SonicG1Core, merged_path: str, dec_path: str) -> bool:
    """End-to-end joint-order sanity: feeding IsaacLab-order obs through the perm graphs
    must equal feeding the equivalent MuJoCo-order obs through the plain core, then the
    action permuted back to IsaacLab. Confirms the bridge is a true inverse pair."""
    print("Perm round-trip (IsaacLab-in perm-graph == MuJoCo-in core, action back to IsaacLab)")
    rng = np.random.default_rng(3)
    B = 8
    # build a 640 encoder input whose joint halves we can permute deterministically
    enc640_isaac = rng.standard_normal((B, SonicG1Core.ENC_INPUT_DIM)).astype(np.float32)
    proprio_isaac = rng.standard_normal((B, SonicG1Core.PROPRIO_DIM)).astype(np.float32)

    enc_sess = ort.InferenceSession(
        os.path.join(os.path.dirname(merged_path), "model_encoder_g1.onnx"),
        providers=["CPUExecutionProvider"],
    )
    dec_sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    tok = _ort_run(enc_sess, enc640_isaac)
    a_graph = _ort_run(dec_sess, np.concatenate([tok, proprio_isaac], axis=1))

    # reference: convert inputs IsaacLab->MuJoCo by hand, run plain core, permute action out
    with torch.no_grad():
        x = torch.from_numpy(enc640_isaac).reshape(B, N_FUT, CMD_PER_FRAME + ANCHOR_PER_FRAME)
        cmd = x[:, :, :CMD_PER_FRAME].reshape(B, N_FUT * 2, NACT)[:, :, IDX_IN].reshape(
            B, N_FUT, CMD_PER_FRAME)
        enc640_m = torch.cat([cmd, x[:, :, CMD_PER_FRAME:]], dim=-1).reshape(
            B, SonicG1Core.ENC_INPUT_DIM)
        pr = torch.from_numpy(proprio_isaac)
        proprio_m = torch.cat([
            pr[:, :_G],
            _perm_blocks(pr[:, _G:_JP], H, IDX_IN),
            _perm_blocks(pr[:, _JP:_JV], H, IDX_IN),
            _perm_blocks(pr[:, _JV:_LA], H, IDX_IN),
            pr[:, _LA:],
        ], dim=1)
        a_ref = core.act_mean(enc640_m, proprio_m)[:, IDX_OUT].numpy()
    return _report("end-to-end perm graph vs core", a_ref, a_graph)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="./last.pt",
                    help="rsl_rl model_*.pt (MuJoCo-order from-scratch) or last.pt (auto-detected).")
    ap.add_argument("--out-dir", default=None, help="output dir (default: <ckpt_dir>/deploy_onnx_perm).")
    ap.add_argument("--merged-1762", action="store_true",
                    help="also export the merged 1762->token encoder as model_encoder.onnx (deploy drop-in).")
    ap.add_argument("--verify", action="store_true", help="verify exported ONNX (export integrity + perm round-trip).")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "deploy_onnx_perm")
    core, fmt = load_core(args.ckpt)
    print(f"Loaded core from: {args.ckpt}\n  format: {fmt}")
    print(f"  FSQ backend: {'official' if core.fsq_is_official else 'FSQFallback'}")
    print(f"  joint bridge: IsaacLab<->MuJoCo baked into encoder/decoder graphs\n")

    enc_g1_path, dec_path = export_split(core, out_dir)
    merged_path = export_merged_encoder(core, out_dir) if args.merged_1762 else None

    if args.verify:
        print()
        ok = verify_self(core, enc_g1_path, dec_path, merged_path)
        if merged_path:
            print()
            ok &= verify_perm_roundtrip(core, merged_path, dec_path)
        print("\n=== SUMMARY ===")
        print(f"  perm-export  {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    if merged_path:
        deployed = "/home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/gear_sonic_deploy/policy/release"
        print(
            f"\nDrop-in install (backs up the originals first):\n"
            f"  cp {deployed}/model_encoder.onnx {deployed}/model_encoder.orig.onnx\n"
            f"  cp {deployed}/model_decoder.onnx {deployed}/model_decoder.orig.onnx\n"
            f"  cp {merged_path} {deployed}/model_encoder.onnx\n"
            f"  cp {dec_path} {deployed}/model_decoder.onnx"
        )


if __name__ == "__main__":
    main()
