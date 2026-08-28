"""
Task-2 verification: exercise the full RSL-RL PPO actor contract on SonicG1ActorModel
WITHOUT needing the env. Mimics exactly what rsl_rl PPO.act / .update / OnPolicyRunner
call on the actor (confirmed against rsl_rl.algorithms.ppo + unilab rsl_rl_ppo).

Run:  uv run --no-sync python scripts/sonic/contract_test.py
      (add --ckpt <last.pt> to also test strict pretrained load)
"""
from __future__ import annotations

import argparse
import io

import torch
from tensordict import TensorDict

from unilab.algos.torch.sonic import SonicG1ActorModel
from unilab.algos.torch.sonic.core import SonicG1Core


def make_obs(batch: int, two_stream: bool, device: str) -> TensorDict:
    enc = torch.randn(batch, SonicG1Core.ENC_INPUT_DIM, device=device)
    proprio = torch.randn(batch, SonicG1Core.PROPRIO_DIM, device=device)
    if two_stream:
        d = {"actor_enc": enc, "actor_proprio": proprio}
    else:
        d = {"actor_enc": torch.cat([enc, proprio], dim=-1)}
    # a critic stream would also be present in the real env; not needed for the actor
    return TensorDict(d, batch_size=[batch], device=device)


def run_contract(actor: SonicG1ActorModel, obs: TensorDict, action_dim: int) -> None:
    # ---- rollout: PPO.act ------------------------------------------------
    actor.get_hidden_state()
    actions = actor(obs, stochastic_output=True)
    assert actions.shape[-1] == action_dim, actions.shape
    logp = actor.get_output_log_prob(actions)
    assert logp.shape == actions.shape[:-1], (logp.shape, actions.shape)
    params = actor.output_distribution_params
    assert isinstance(params, tuple) and len(params) == 2
    actor.update_normalization(obs)
    actor.reset(torch.zeros(obs.batch_size[0], dtype=torch.bool, device=obs.device))

    # ---- deterministic path (eval / mean) --------------------------------
    mean = actor(obs, stochastic_output=False)
    assert torch.allclose(mean, actor.output_mean)
    _ = actor.output_std
    _ = actor.output_entropy

    # ---- update: recompute + KL + backward -------------------------------
    old_params = tuple(p.detach() for p in params)
    actor(obs, stochastic_output=True)
    new_logp = actor.get_output_log_prob(actions.detach())
    entropy = actor.output_entropy
    kl = actor.get_kl_divergence(old_params, actor.output_distribution_params)
    assert kl.shape[0] == obs.batch_size[0], kl.shape
    loss = -(new_logp.mean()) - 0.01 * entropy.mean()
    loss.backward()
    grads = [p.grad for p in actor.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients flowed to trainable params"
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)

    # ---- runner logging + export -----------------------------------------
    _ = actor.output_std  # OnPolicyRunner logs this each iter
    onnx_mod = actor.as_onnx(verbose=False).to("cpu")  # export/deploy is CPU/NPU
    dummy = onnx_mod.get_dummy_inputs()
    with torch.no_grad():
        ref = onnx_mod(*dummy)
    assert ref.shape == (1, action_dim), ref.shape
    buf = io.BytesIO()
    torch.onnx.export(
        onnx_mod, dummy, buf,
        input_names=onnx_mod.input_names, output_names=onnx_mod.output_names,
        opset_version=18, dynamic_axes={n: {0: "batch"} for n in onnx_mod.input_names},
    )
    # numeric parity onnx vs torch
    try:
        import numpy as np
        import onnxruntime as ort

        buf.seek(0)
        sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
        feeds = {n: d.numpy() for n, d in zip(onnx_mod.input_names, dummy)}
        out = sess.run(None, feeds)[0]
        md = float(np.abs(out - ref.numpy()).max())
        assert md < 1e-4, f"onnx parity max_diff={md}"
        print(f"    onnx parity max_diff={md:.2e} (<1e-4) OK")
    except ImportError:
        print("    onnxruntime not installed; skipped parity check (export itself OK)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="sonic_release/last.pt for strict-load test")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = a.device
    B, ACT = 8, SonicG1Core.ACTION_DIM

    print("[1] two-stream, FSQ on, scalar std")
    actor = SonicG1ActorModel(
        make_obs(B, True, dev), {"actor": ["actor_enc"]}, "actor", ACT,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ).to(dev)
    run_contract(actor, make_obs(B, True, dev), ACT)
    print(f"    FSQ backend official={actor.core.fsq_is_official}")

    print("[2] single-stream split fallback (proprio_group=None)")
    actor2 = SonicG1ActorModel(
        make_obs(B, False, dev), {"actor": ["actor_enc"]}, "actor", ACT,
        proprio_group=None,
    ).to(dev)
    run_contract(actor2, make_obs(B, False, dev), ACT)

    print("[3] obs_normalization=True, log std")
    actor3 = SonicG1ActorModel(
        make_obs(B, True, dev), {"actor": ["actor_enc"]}, "actor", ACT,
        obs_normalization=True, distribution_cfg={"init_std": 0.5, "std_type": "log"},
    ).to(dev)
    run_contract(actor3, make_obs(B, True, dev), ACT)

    print("[4] no-FSQ baseline (task 5) + kin aux + freeze encoder")
    actor4 = SonicG1ActorModel(
        make_obs(B, True, dev), {"actor": ["actor_enc"]}, "actor", ACT,
        use_fsq=False, with_kin_aux=True, freeze_encoder=True,
    ).to(dev)
    run_contract(actor4, make_obs(B, True, dev), ACT)
    recon = actor4.kin_recon(make_obs(B, True, dev))
    assert recon is not None and recon.shape == (B, SonicG1Core.ENC_INPUT_DIM), recon.shape
    enc_params = list(actor4.core.encoder.parameters())
    assert all(not p.requires_grad for p in enc_params), "encoder should be frozen"
    print("    kin aux recon + frozen-encoder OK")

    if a.ckpt:
        print("[5] strict pretrained load from last.pt")
        actor5 = SonicG1ActorModel(
            make_obs(B, True, dev), {"actor": ["actor_enc"]}, "actor", ACT,
            pretrained_ckpt=a.ckpt,
        ).to(dev)
        run_contract(actor5, make_obs(B, True, dev), ACT)
        print("    strict load OK")

    print("\nALL CONTRACT CHECKS PASSED")


if __name__ == "__main__":
    main()
