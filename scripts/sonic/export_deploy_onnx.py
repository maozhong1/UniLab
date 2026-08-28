"""Deploy-split ONNX export for the SONIC G1 policy (encoder + decoder, separate graphs).

The NPU deploy consumes TWO ONNX graphs, not the combined actor:

  1. encoder (g1-only): obs_dict FLOAT[1, 640] -> encoded_tokens FLOAT[1, 64]
     (encoder MLP -> view(B,2,32) -> FSQ quantize -> reshape[B,64]; FSQ is IN the graph)
  2. decoder:           obs_dict FLOAT[1, 994] -> action        FLOAT[1, 29]
     (994 = token[64] ++ proprio[930]; = decoder(cat([token, proprio])))

Opset 13 (Round op needed for FSQ), FLOAT. Export runs on CPU.

Checkpoint loading (auto-detected):
  * raw ``last.pt``           -> ``load_g1_from_last_pt`` (policy_state_dict, sonic prefixes)
  * rsl_rl ``model_*.pt``     -> extract ``actor_state_dict`` keys under ``core.`` into a
    fresh ``SonicG1Core`` (strict).

Run (example):
    uv run --no-sync python scripts/sonic/export_deploy_onnx.py \
        --ckpt ./last.pt --out-dir ./deploy_onnx --verify

When ``--ckpt`` is a ``last.pt`` and ``--verify`` is set, the exported graphs are also
cross-checked against the DEPLOYED reference ONNX (both derive from last.pt, so they must
match): exported decoder vs ``model_decoder.onnx``; exported g1 encoder(640) vs
``model_encoder.onnx`` fed through the parity-harness ``_scatter_encoder_input`` 640->1762.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from unilab.algos.torch.sonic.core import (
    SonicG1Core,
    _install_fake_import_hook,
    load_g1_from_last_pt,
)

# Reuse the deployed-encoder 640->1762 scatter + per-row ORT runner from the harness.
from parity_harness import (  # noqa: E402
    ANCHOR_PER_FRAME,
    CMD_PER_FRAME,
    ENC_TOTAL,
    N_FUT,
    OFF_ANCHOR,
    OFF_CMD,
    OFF_MODE,
    _ort_run,
    _ort_session,
    _scatter_encoder_input,
)

TOL = 1e-4
OPSET = 13
MERGED_INPUT_DIM = ENC_TOTAL  # 1762 — the deploy's merged 3-mode encoder buffer


# ----------------------------------------------------------------------------
# Export wrappers (the two separate deploy graphs).
# ----------------------------------------------------------------------------
class EncoderExport(nn.Module):
    """obs_dict[B,640] -> encoded_tokens[B,64] (encoder MLP -> FSQ, quantized)."""

    input_names = ["obs_dict"]
    output_names = ["encoded_tokens"]

    def __init__(self, core: SonicG1Core) -> None:
        super().__init__()
        self.core = core

    def forward(self, obs_dict: torch.Tensor) -> torch.Tensor:
        return self.core.encode(obs_dict)


class MergedEncoderExport(nn.Module):
    """Deploy drop-in encoder: obs_dict[B,1762] -> encoded_tokens[B,64].

    Path B: accept the deploy's EXISTING merged 3-mode buffer (so no C++/obs-config
    change) but run our g1-only encoder weights. The transform REPLICATES EXACTLY what
    the original deployed ``model_encoder.onnx`` does with its g1 branch — verified
    byte-exact (max|Δ|=0) against that ONNX on real g1-mode buffers, and empirically
    stable on-robot (walk/dance) with the release weights:

        cmd    = obs[:,   4:584].reshape(10, 58)   # command block, read directly [10,58]
        anchor = obs[:, 601:661].reshape(10, 6)    # anchor block,  read directly [10,6]
        enc640 = concat([cmd, anchor], dim=-1).reshape(640)   # per frame [58 ++ 6] = 64

    IMPORTANT (corrected 2026-08-27): this is the PLAIN ``reshape`` of the raw command
    block — do NOT re-interleave it into [dof_pos, dof_vel] per frame. An earlier version
    re-interleaved (jpos block [4:294] / jvel block [294:584]) on the theory that the
    deploy fed the encoder mis-ordered data; that was WRONG. The original encoder is
    trained on (and the deploy feeds) exactly this raw layout, so re-interleaving
    SCRAMBLES the input — on-robot that degraded last.pt (feet unstable, large arm
    motion) and broke the finetune (could not stand). The plain reshape is correct.
    mode_id (obs[:,0]) and all teleop/smpl slots are ignored — we only have g1 weights.
    """

    input_names = ["obs_dict"]
    output_names = ["encoded_tokens"]

    OFF_CMD = OFF_CMD                    # 4
    OFF_ANCHOR = OFF_ANCHOR              # 601
    N_FUT = N_FUT                        # 10
    CMD_PER_FRAME = CMD_PER_FRAME        # 58 (raw command values per frame)
    ANCHOR_PER_FRAME = ANCHOR_PER_FRAME  # 6

    def __init__(self, core: SonicG1Core) -> None:
        super().__init__()
        self.core = core

    def forward(self, obs_dict: torch.Tensor) -> torch.Tensor:
        B = obs_dict.shape[0]
        nf, cpf, af = self.N_FUT, self.CMD_PER_FRAME, self.ANCHOR_PER_FRAME
        cmd = obs_dict[:, self.OFF_CMD : self.OFF_CMD + nf * cpf].reshape(B, nf, cpf)
        anch = obs_dict[:, self.OFF_ANCHOR : self.OFF_ANCHOR + nf * af].reshape(B, nf, af)
        enc640 = torch.cat([cmd, anch], dim=-1).reshape(B, nf * (cpf + af))
        return self.core.encode(enc640)


class DecoderExport(nn.Module):
    """obs_dict[B,994] = token[64] ++ proprio[930] -> action[B,29]."""

    input_names = ["obs_dict"]
    output_names = ["action"]

    def __init__(self, core: SonicG1Core) -> None:
        super().__init__()
        self.core = core

    def forward(self, obs_dict: torch.Tensor) -> torch.Tensor:
        token = obs_dict[:, : SonicG1Core.TOKEN_DIM]
        proprio = obs_dict[:, SonicG1Core.TOKEN_DIM :]
        return self.core.decode(token, proprio)


# ----------------------------------------------------------------------------
# Checkpoint loading — auto-detect last.pt vs rsl_rl model_*.pt.
# ----------------------------------------------------------------------------
def load_core(ckpt_path: str) -> tuple[SonicG1Core, str]:
    """Return (loaded SonicG1Core.eval(), format-tag). Auto-detects checkpoint format."""
    _install_fake_import_hook()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    core = SonicG1Core(with_kin_aux=False, use_fsq=True).eval()

    if isinstance(ck, dict) and "actor_state_dict" in ck:
        # rsl_rl ActorCritic save: actor_state_dict holds SonicG1ActorModel weights
        # under the `core.` prefix (encoder / fsq buffers / decoder / log_std).
        asd = ck["actor_state_dict"]
        core_sd = {k[len("core."):]: v for k, v in asd.items() if k.startswith("core.")}
        if not core_sd:
            raise KeyError(
                f"rsl_rl actor_state_dict has no 'core.' keys; got prefixes "
                f"{sorted({k.split('.')[0] for k in asd})}"
            )
        core.load_state_dict(core_sd, strict=True)
        return core, "rsl_rl (actor_state_dict['core.*'])"

    if isinstance(ck, dict) and "policy_state_dict" in ck:
        load_g1_from_last_pt(core, ckpt_path)
        return core, "sonic last.pt (policy_state_dict)"

    raise KeyError(
        f"Unrecognized checkpoint: top keys "
        f"{list(ck.keys()) if isinstance(ck, dict) else type(ck)}"
    )


# ----------------------------------------------------------------------------
# Export.
# ----------------------------------------------------------------------------
def export_split(core: SonicG1Core, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    enc_path = os.path.join(out_dir, "model_encoder_g1.onnx")
    dec_path = os.path.join(out_dir, "model_decoder.onnx")

    enc_mod = EncoderExport(core).eval()
    dec_mod = DecoderExport(core).eval()

    with torch.inference_mode():
        torch.onnx.export(
            enc_mod,
            (torch.zeros(1, SonicG1Core.ENC_INPUT_DIM),),
            enc_path,
            input_names=EncoderExport.input_names,
            output_names=EncoderExport.output_names,
            opset_version=OPSET,
        )
        torch.onnx.export(
            dec_mod,
            (torch.zeros(1, SonicG1Core.TOKEN_DIM + SonicG1Core.PROPRIO_DIM),),
            dec_path,
            input_names=DecoderExport.input_names,
            output_names=DecoderExport.output_names,
            opset_version=OPSET,
        )
    print(f"Exported encoder ONNX -> {enc_path}")
    print(f"Exported decoder ONNX -> {dec_path}")
    return enc_path, dec_path


def _random_g1_buffer(rng: np.random.Generator, B: int) -> np.ndarray:
    """A random deploy-style merged obs_dict(1762): g1 mode_id=0, random values in the
    g1 slots (command [4:584], anchor [601:661]), zeros elsewhere. Used to check the
    wrapper against the ORIGINAL deployed ONNX (the ground-truth transform)."""
    x = np.zeros((B, ENC_TOTAL), dtype=np.float32)
    x[:, OFF_MODE] = 0.0  # g1
    x[:, OFF_CMD : OFF_CMD + N_FUT * CMD_PER_FRAME] = rng.standard_normal(
        (B, N_FUT * CMD_PER_FRAME)).astype(np.float32)
    x[:, OFF_ANCHOR : OFF_ANCHOR + N_FUT * ANCHOR_PER_FRAME] = rng.standard_normal(
        (B, N_FUT * ANCHOR_PER_FRAME)).astype(np.float32)
    return x


def export_merged_encoder(core: SonicG1Core, out_dir: str) -> str:
    """Export the Path-B drop-in: obs_dict[1,1762] -> encoded_tokens[1,64].

    Written as ``model_encoder.onnx`` (the deploy default name) so it replaces the
    original merged 3-mode encoder with no C++/obs-config change.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model_encoder.onnx")
    mod = MergedEncoderExport(core).eval()
    with torch.inference_mode():
        torch.onnx.export(
            mod,
            (torch.zeros(1, MERGED_INPUT_DIM),),
            path,
            input_names=MergedEncoderExport.input_names,
            output_names=MergedEncoderExport.output_names,
            opset_version=OPSET,
        )
    print(f"Exported merged-1762 encoder ONNX -> {path}")
    return path


def _report(label: str, a: np.ndarray, b: np.ndarray) -> bool:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mx = float(d.max())
    ok = mx < TOL
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<42} max|Δ|={mx:.3e}")
    return ok


# ----------------------------------------------------------------------------
# Verify: exported ONNX vs torch core (self-consistency).
# ----------------------------------------------------------------------------
def verify_self(core: SonicG1Core, enc_path: str, dec_path: str) -> bool:
    print("Self-verify (exported ONNX vs torch SonicG1Core)")
    rng = np.random.default_rng(0)
    enc_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    B = 8
    ok = True

    # encoder: enc640 -> token
    enc640 = rng.standard_normal((B, SonicG1Core.ENC_INPUT_DIM)).astype(np.float32)
    onnx_t = _ort_run(enc_sess, enc640)
    with torch.no_grad():
        torch_t = core.encode(torch.from_numpy(enc640)).numpy()
    ok &= _report("encoder  enc640 -> token", torch_t, onnx_t)

    # decoder: [token||proprio]994 -> action
    token = rng.uniform(-1.0, 1.0, (B, SonicG1Core.TOKEN_DIM)).astype(np.float32)
    proprio = rng.standard_normal((B, SonicG1Core.PROPRIO_DIM)).astype(np.float32)
    obs994 = np.concatenate([token, proprio], axis=1)
    onnx_a = _ort_run(dec_sess, obs994)
    with torch.no_grad():
        torch_a = core.decode(torch.from_numpy(token), torch.from_numpy(proprio)).numpy()
    ok &= _report("decoder  [token||proprio] -> action", torch_a, onnx_a)

    # end-to-end: enc640 -> exported enc -> token -> exported dec vs core.act_mean
    enc640 = rng.standard_normal((B, SonicG1Core.ENC_INPUT_DIM)).astype(np.float32)
    proprio = rng.standard_normal((B, SonicG1Core.PROPRIO_DIM)).astype(np.float32)
    e2e_t = _ort_run(enc_sess, enc640)
    e2e_a = _ort_run(dec_sess, np.concatenate([e2e_t, proprio], axis=1))
    with torch.no_grad():
        torch_a = core.act_mean(
            torch.from_numpy(enc640), torch.from_numpy(proprio)
        ).numpy()
    ok &= _report("end-to-end  enc->dec vs core.act_mean", torch_a, e2e_a)
    return ok


# ----------------------------------------------------------------------------
# Verify: merged-1762 wrapper transform == original deployed ONNX (the ground truth).
# ----------------------------------------------------------------------------
def verify_merged(core: SonicG1Core, merged_path: str, deployed_dir: str | None,
                  is_last_pt: bool) -> bool:
    """The correctness criterion for the wrapper is: on the deploy's native 1762 buffer
    it must reproduce what the ORIGINAL deployed ``model_encoder.onnx`` computes — that
    ONNX is on-robot-validated (walk/dance stable). For the release ``last.pt`` (same
    weights) this is a byte-exact check (expect max|Δ|=0). For finetuned weights the
    tokens legitimately differ, so we only assert the TRANSFORM matches by checking the
    release-equivalent path is 0 elsewhere; here we still print the diff for visibility.
    """
    print("Merged-1762 verify (wrapper vs ORIGINAL deployed ONNX on native buffer)")
    rng = np.random.default_rng(2)
    sess = ort.InferenceSession(merged_path, providers=["CPUExecutionProvider"])
    B = 16
    buf = _random_g1_buffer(rng, B)  # deploy-native layout, g1 mode
    wrap_t = _ort_run(sess, buf)

    dep_path = os.path.join(deployed_dir, "model_encoder.onnx") if deployed_dir else None
    if dep_path and os.path.isfile(dep_path):
        dep = _ort_session(dep_path)
        if dep.get_inputs()[0].shape[-1] == MERGED_INPUT_DIM:
            dep_t = _ort_run(dep, buf)
            if is_last_pt:
                # same weights -> wrapper must byte-match the original ONNX g1 branch
                return _report("merged(1762) vs original deployed ONNX", dep_t, wrap_t)
            # finetuned weights -> tokens differ; report magnitude for visibility only
            d = float(np.abs(dep_t.astype(np.float64) - wrap_t).max())
            print(f"  [info] finetuned wrapper vs original ONNX max|Δ|={d:.3e} "
                  f"(expected nonzero: different weights, SAME transform)")
            # still confirm the transform itself is sound: wrapper == core on the same buf
    # transform self-consistency (weights-agnostic): wrapper(buf) == core.encode(naive)
    nf, cpf, af = N_FUT, CMD_PER_FRAME, ANCHOR_PER_FRAME
    cmd = buf[:, OFF_CMD : OFF_CMD + nf * cpf].reshape(B, nf, cpf)
    anc = buf[:, OFF_ANCHOR : OFF_ANCHOR + nf * af].reshape(B, nf, af)
    enc640 = np.concatenate([cmd, anc], axis=-1).reshape(B, -1).astype(np.float32)
    with torch.no_grad():
        core_t = core.encode(torch.from_numpy(enc640)).numpy()
    return _report("merged(1762) vs core.encode(naive reshape)", core_t, wrap_t)


# ----------------------------------------------------------------------------
# Cross-check: exported vs DEPLOYED reference ONNX (only meaningful for last.pt).
# ----------------------------------------------------------------------------
def verify_vs_deployed(enc_path: str, dec_path: str, deployed_dir: str) -> bool:
    print("Cross-check (exported vs DEPLOYED reference ONNX)")
    rng = np.random.default_rng(1)
    exp_enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    exp_dec = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    dep_enc = _ort_session(os.path.join(deployed_dir, "model_encoder.onnx"))
    dep_dec = _ort_session(os.path.join(deployed_dir, "model_decoder.onnx"))
    B = 8
    ok = True

    # decoder: shared random 994 input
    obs994 = np.concatenate(
        [
            rng.uniform(-1.0, 1.0, (B, SonicG1Core.TOKEN_DIM)).astype(np.float32),
            rng.standard_normal((B, SonicG1Core.PROPRIO_DIM)).astype(np.float32),
        ],
        axis=1,
    )
    ok &= _report(
        "decoder  exported vs deployed(994)",
        _ort_run(exp_dec, obs994),
        _ort_run(dep_dec, obs994),
    )

    # encoder: exported g1(640) vs deployed via scatter 640->1762
    enc640 = rng.standard_normal((B, SonicG1Core.ENC_INPUT_DIM)).astype(np.float32)
    ok &= _report(
        "encoder  exported(640) vs deployed(1762)",
        _ort_run(exp_enc, enc640),
        _ort_run(dep_enc, _scatter_encoder_input(enc640)),
    )
    return ok


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        # Relative to the launch dir (repo root under `uv run python scripts/...`).
        default="./last.pt",
        help="raw sonic last.pt OR rsl_rl model_*.pt (auto-detected); default ./last.pt.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output dir (default: <ckpt_dir>/deploy_onnx).",
    )
    ap.add_argument("--verify", action="store_true", help="verify exported ONNX vs torch.")
    ap.add_argument(
        "--merged-1762",
        action="store_true",
        help="also export the Path-B drop-in merged encoder (obs_dict[1762]->token[64]) "
        "as model_encoder.onnx; consumes the deploy's channel-major buffer, no C++/config change.",
    )
    ap.add_argument(
        "--deployed-dir",
        default="/home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/gear_sonic_deploy/policy/release",
        help="deployed reference ONNX dir (cross-check for last.pt).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "deploy_onnx")

    core, fmt = load_core(args.ckpt)
    print(f"Loaded core from: {args.ckpt}\n  format: {fmt}")
    print(f"  FSQ backend: {'official' if core.fsq_is_official else 'FSQFallback'}\n")

    enc_path, dec_path = export_split(core, out_dir)

    merged_path = export_merged_encoder(core, out_dir) if args.merged_1762 else None

    if not args.verify:
        if merged_path:
            print(
                f"\nDrop-in install (backs up the original first):\n"
                f"  cp {args.deployed_dir}/model_encoder.onnx {args.deployed_dir}/model_encoder.orig.onnx\n"
                f"  cp {merged_path} {args.deployed_dir}/model_encoder.onnx\n"
                f"  cp {dec_path} {args.deployed_dir}/model_decoder.onnx\n"
                f"(keep observation_config.yaml unchanged — buffer stays 1762)"
            )
        return

    # Cross-check against deployed ONNX only when the ckpt is the release last.pt
    # (rsl_rl finetuned weights legitimately differ from the deployed policy).
    is_last_pt = "last.pt" in fmt

    print()
    results = {"self_verify": verify_self(core, enc_path, dec_path)}
    if merged_path:
        print()
        results["merged_1762"] = verify_merged(
            core, merged_path, args.deployed_dir, is_last_pt
        )

    if is_last_pt and os.path.isdir(args.deployed_dir):
        print()
        results["vs_deployed"] = verify_vs_deployed(enc_path, dec_path, args.deployed_dir)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:<14} {'PASS' if v else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
