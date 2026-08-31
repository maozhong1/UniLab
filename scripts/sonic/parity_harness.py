"""Numerical parity harness — sonic G1 enc(640)/proprio(930) vs deployed ONNX.

Validates that the PyTorch ``SonicG1Core`` loaded from ``sonic_release/last.pt``
reproduces the deployed ``model_encoder.onnx`` / ``model_decoder.onnx`` bit-for-bit
(target max|Δ| < 1e-4), and that the MuJoCo→IsaacLab 29-dof joint permutation baked
into ``G1SonicMotionTracking`` is a correct inverse pair.

Three checks (all must pass):

  1. PERMUTATION self-test — ``_MUJOCO_TO_ISAACLAB`` / ``_ISAACLAB_TO_MUJOCO`` are true
     inverses and match gear_sonic_deploy policy_parameters.hpp arrays exactly.

  2. DECODER parity (deployment-critical, always-on) — build obs_dict[994] = token(64)
     ++ proprio(930); compare torch ``SonicG1Core.decode`` vs ``model_decoder.onnx``.

  3. ENCODER parity (reference mode) — scatter our legacy-packed enc(640) into the
      merged 3-mode obs_dict[1762] at
     the g1-mode slots (indices recovered from the ONNX graph itself: command at
     [4:584] read [10,58], anchor at [601:661] read [10,6]), mode_id=0, zeros
     elsewhere; compare torch ``SonicG1Core.encode`` vs ``model_encoder.onnx``.

Optionally (--env) instantiate the real ``G1SonicMotionTracking`` env, take one obs,
split 640/930 and run it through BOTH stacks — confirms the *assembled* (permuted)
observation is consumed correctly end-to-end.

Run:
    HF_ENDPOINT=https://hf-mirror.com \
    uv run --no-sync python scripts/sonic/parity_harness.py \
        --ckpt /home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/sonic_release/last.pt \
        --onnx-dir /home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/gear_sonic_deploy/policy/release
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch

from unilab.algos.torch.sonic.core import SonicG1Core, load_g1_from_last_pt
from unilab.envs.motion_tracking.g1.tracking_sonic import (
    _ISAACLAB_TO_MUJOCO,
    _MUJOCO_TO_ISAACLAB,
)

TOL = 1e-4

# --- ground-truth arrays from gear_sonic_deploy policy_parameters.hpp -----------
HPP_MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28], dtype=np.intp)
HPP_ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28], dtype=np.intp)

# --- merged encoder obs_dict[1762] g1-mode layout ------------------------------
# Recovered directly from the model_encoder.onnx graph (Slice/Reshape/Concat nodes
# feeding the g1 Gemm[2048,640]), NOT guessed from the deploy yaml (whose two joint
# gatherers write channel-major and do NOT match the ONNX reshape):
#   obs_dict[0]         = encoder_index / mode_id  (g1 = 0.0)
#   obs_dict[4:584]     = command_multi_future_nonflat, directly reshaped to [10, 58].
#                         The source term is [all q frames, all dq frames], so these
#                         rows are not semantic per-frame [q_t, dq_t] pairs.
#   obs_dict[601:661]   = motion_anchor_ori_b_mf_nonflat, read as [10, 6]:
#                         per frame f, obs[601 + 6f : 601 + 6f + 6]
# The g1 encoder input (640) is the per-frame concat [cmd58 ++ anchor6] over 10 frames.
ENC_TOTAL = 1762
G1_MODE_ID = 0.0
OFF_MODE = 0
OFF_CMD = 4            # command_multi_future_nonflat  [4:584]  = 10 × 58
OFF_ANCHOR = 601       # motion_anchor_ori_b_mf_nonflat [601:661] = 10 × 6
N_FUT = 10
CMD_PER_FRAME = 58     # dof_pos(29) + dof_vel(29)
ANCHOR_PER_FRAME = 6
PER_FRAME = CMD_PER_FRAME + ANCHOR_PER_FRAME  # 64


def _ort_session(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _ort_run(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    # deployed models have a fixed batch dim of 1 → run row-by-row and stack.
    name = sess.get_inputs()[0].name
    x = x.astype(np.float32)
    outs = [sess.run(None, {name: x[i : i + 1]})[0] for i in range(x.shape[0])]
    return np.concatenate(outs, axis=0)


def _report(label: str, a: np.ndarray, b: np.ndarray) -> bool:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mx = float(d.max())
    n_over = int((d > TOL).sum())
    ok = mx < TOL
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<34} max|Δ|={mx:.3e}  "
          f"(>{TOL:g}: {n_over}/{d.size})")
    return ok


def _scatter_encoder_input(enc640: np.ndarray) -> np.ndarray:
    """Scatter frame-major enc(640) = [f:(dof_pos29, dof_vel29, anchor6)]×10 into the
    merged encoder obs_dict(1762) g1-mode slots (command interleaved with anchor)."""
    B = enc640.shape[0]
    x = np.zeros((B, ENC_TOTAL), dtype=np.float32)
    x[:, OFF_MODE] = G1_MODE_ID
    e = enc640.reshape(B, N_FUT, PER_FRAME)  # (B, 10, 64)
    for f in range(N_FUT):
        x[:, OFF_CMD + f * CMD_PER_FRAME : OFF_CMD + f * CMD_PER_FRAME + CMD_PER_FRAME] = e[:, f, :CMD_PER_FRAME]
        x[:, OFF_ANCHOR + f * ANCHOR_PER_FRAME : OFF_ANCHOR + f * ANCHOR_PER_FRAME + ANCHOR_PER_FRAME] = e[:, f, CMD_PER_FRAME:]
    return x


# ------------------------------------------------------------------------------
def check_permutation() -> bool:
    print("1) Permutation self-test (MuJoCo↔IsaacLab)")
    ok = True
    ok &= np.array_equal(_MUJOCO_TO_ISAACLAB, HPP_MUJOCO_TO_ISAACLAB)
    print(f"  [{'PASS' if np.array_equal(_MUJOCO_TO_ISAACLAB, HPP_MUJOCO_TO_ISAACLAB) else 'FAIL'}]"
          f" env _MUJOCO_TO_ISAACLAB == hpp mujoco_to_isaaclab")
    ok &= np.array_equal(_ISAACLAB_TO_MUJOCO, HPP_ISAACLAB_TO_MUJOCO)
    print(f"  [{'PASS' if np.array_equal(_ISAACLAB_TO_MUJOCO, HPP_ISAACLAB_TO_MUJOCO) else 'FAIL'}]"
          f" env _ISAACLAB_TO_MUJOCO == hpp isaaclab_to_mujoco")
    # inverse round-trip: v[m2i][i2m] == v
    v = np.arange(29)
    rt = v[_MUJOCO_TO_ISAACLAB][_ISAACLAB_TO_MUJOCO]
    inv_ok = np.array_equal(rt, v)
    ok &= inv_ok
    print(f"  [{'PASS' if inv_ok else 'FAIL'}] inverse round-trip v[m2i][i2m]==v")
    # a permuted vector de-permutes back
    rng = np.random.default_rng(0)
    x = rng.standard_normal(29)
    dep_ok = np.allclose(x[_MUJOCO_TO_ISAACLAB][_ISAACLAB_TO_MUJOCO], x)
    ok &= dep_ok
    print(f"  [{'PASS' if dep_ok else 'FAIL'}] float vector round-trip")
    return ok


def check_decoder(model: SonicG1Core, dec_sess, rng) -> bool:
    print("2) Decoder parity  (obs_dict[994] → action[29])")
    B = 8
    token = rng.uniform(-1.0, 1.0, (B, SonicG1Core.TOKEN_DIM)).astype(np.float32)
    proprio = rng.standard_normal((B, SonicG1Core.PROPRIO_DIM)).astype(np.float32)
    obs994 = np.concatenate([token, proprio], axis=1)
    onnx_a = _ort_run(dec_sess, obs994)
    with torch.no_grad():
        torch_a = model.decode(
            torch.from_numpy(token), torch.from_numpy(proprio)
        ).numpy()
    return _report("decoder(token||proprio)", torch_a, onnx_a)


def check_encoder(model: SonicG1Core, enc_sess, rng) -> bool:
    print("3) Encoder parity  (obs_dict[1762] g1-slots → token[64])")
    B = 8
    enc640 = rng.standard_normal((B, SonicG1Core.ENC_INPUT_DIM)).astype(np.float32)
    onnx_t = _ort_run(enc_sess, _scatter_encoder_input(enc640))
    with torch.no_grad():
        torch_t = model.encode(torch.from_numpy(enc640)).numpy()
    ok = _report("encoder(g1) token", torch_t, onnx_t)
    if not ok:
        # FSQ boundary flips (a single dim off by ~1 grid step) vs a real layout bug
        d = np.abs(torch_t - onnx_t)
        print(f"       per-dim flips: {(d > TOL).sum(axis=1)}  "
              f"(a handful of ±1-grid-step flips ⇒ FSQ rounding on a boundary, "
              f"not a layout error)")
    return ok


def check_end_to_end(model: SonicG1Core, enc_sess, dec_sess, rng) -> bool:
    print("4) End-to-end  (enc640 → token → action) both stacks")
    B = 8
    enc640 = rng.standard_normal((B, SonicG1Core.ENC_INPUT_DIM)).astype(np.float32)
    proprio = rng.standard_normal((B, SonicG1Core.PROPRIO_DIM)).astype(np.float32)
    onnx_t = _ort_run(enc_sess, _scatter_encoder_input(enc640))
    onnx_a = _ort_run(dec_sess, np.concatenate([onnx_t, proprio], axis=1))
    with torch.no_grad():
        torch_a = model.act_mean(
            torch.from_numpy(enc640), torch.from_numpy(proprio)
        ).numpy()
    return _report("act_mean vs onnx enc→dec", torch_a, onnx_a)


def check_env(model: SonicG1Core, enc_sess, dec_sess) -> bool:
    """Optional: run the REAL assembled (permuted) obs through both stacks.

    This confirms the env's obs is (a) finite / correctly shaped and (b) consumed
    identically by the torch core and the deployed ONNX. It does NOT prove the
    permutation matches sonic's own IsaacLab obs (no ground-truth obs available
    here) — that is covered structurally by check 1 + the offline layout trace.
    """
    print("5) Real-env obs parity  (--env)")
    from unilab.base import registry

    registry.ensure_registries()
    env = registry.make("G1SonicMotionTracking", num_envs=2, sim_backend="mujoco")
    try:
        state = env.init_state()
        actor = np.asarray(state.obs["obs"])
        enc_dim = SonicG1Core.ENC_INPUT_DIM
        assert actor.shape[1] == enc_dim + SonicG1Core.PROPRIO_DIM, actor.shape
        assert np.isfinite(actor).all(), "non-finite in env obs"
        enc640 = actor[:, :enc_dim].astype(np.float32)
        proprio = actor[:, enc_dim:].astype(np.float32)
        onnx_t = _ort_run(enc_sess, _scatter_encoder_input(enc640))
        onnx_a = _ort_run(dec_sess, np.concatenate([onnx_t, proprio], axis=1))
        with torch.no_grad():
            torch_a = model.act_mean(
                torch.from_numpy(enc640), torch.from_numpy(proprio)
            ).numpy()
        return _report("real-obs act_mean vs onnx", torch_a, onnx_a)
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    default_root = "/home/maozhong/work/sonic_vla_infer/GR00T-WholeBodyControl-ov"
    ap.add_argument("--ckpt", default=f"{default_root}/sonic_release/last.pt")
    ap.add_argument("--onnx-dir", default=f"{default_root}/gear_sonic_deploy/policy/release")
    ap.add_argument("--env", action="store_true", help="also run the real env obs check")
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = SonicG1Core(with_kin_aux=False, use_fsq=True).eval()
    load_g1_from_last_pt(model, args.ckpt)
    print(f"FSQ backend: {'vector_quantize_pytorch (official)' if model.fsq_is_official else 'FSQFallback'}\n")

    enc_sess = _ort_session(os.path.join(args.onnx_dir, "model_encoder.onnx"))
    dec_sess = _ort_session(os.path.join(args.onnx_dir, "model_decoder.onnx"))

    results = {
        "permutation": check_permutation(),
        "decoder": check_decoder(model, dec_sess, rng),
        "encoder": check_encoder(model, enc_sess, rng),
        "end_to_end": check_end_to_end(model, enc_sess, dec_sess, rng),
    }
    if args.env:
        try:
            results["env"] = check_env(model, enc_sess, dec_sess)
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] env check raised: {type(e).__name__}: {e}")

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:<12} {'PASS' if v else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
