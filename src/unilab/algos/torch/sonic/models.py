"""
UniLab RSL-RL ActorCritic wrapper around ``SonicG1Core`` (task 2 integration).

Design (mirrors HORA's ``HoraActorModel`` extension pattern in ``../hora/models.py``):

* The actor is a *custom* model plugged via ``class_name: unilab.algos.torch.sonic:SonicG1ActorModel``.
* Unlike a plain MLP it needs **two** separate obs streams kept apart:
  the sonic G1-encoder reference input (``enc_dim=640``) and the decoder
  proprioception (``proprio_dim=930``). It therefore indexes named TensorDict keys
  (like HORA indexes ``obs["actor"]`` / ``obs["priv_info"]``) rather than concatenating
  ``obs_groups[obs_set]`` into a single latent.
* The action std is owned by an ``rsl_rl.modules.GaussianDistribution`` (the RL policy
  std the PPO runner reads), NOT by ``SonicG1Core.log_std`` (which is only informational
  / can seed ``init_std``).
* The **critic** deliberately stays a stock ``rsl_rl.models.MLPModel`` (configured in
  YAML) so the value net is unquantized and expressive; nothing to implement here.

obs routing:
  - two-stream (recommended, task 3 emits both): ``enc_group`` and ``proprio_group``.
  - single-stream fallback: set ``proprio_group=null``; the ``enc_group`` tensor of
    width ``enc_dim+proprio_dim`` is split internally.
"""
from __future__ import annotations

import copy
from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization, GaussianDistribution
from tensordict import TensorDict

from .core import SonicG1Core, load_g1_from_last_pt


class _SonicInferenceModule(nn.Module):
    """Exportable deterministic actor: (enc, proprio) -> action. Used by as_jit/as_onnx."""

    input_names = ["actor_enc", "actor_proprio"]
    output_names = ["actions"]

    def __init__(
        self,
        *,
        core: SonicG1Core,
        obs_normalizer: nn.Module,
        enc_dim: int,
        proprio_dim: int,
    ) -> None:
        super().__init__()
        self.core = core
        self.obs_normalizer = obs_normalizer
        self.enc_dim = int(enc_dim)
        self.proprio_dim = int(proprio_dim)

    def forward(self, actor_enc: torch.Tensor, actor_proprio: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.obs_normalizer, nn.Identity):
            x = self.obs_normalizer(torch.cat([actor_enc, actor_proprio], dim=-1))
            actor_enc = x[..., : self.enc_dim]
            actor_proprio = x[..., self.enc_dim :]
        return self.core.act_mean(actor_enc, actor_proprio)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(1, self.enc_dim), torch.zeros(1, self.proprio_dim))


class SonicG1ActorModel(nn.Module):
    """RSL-RL actor wrapping the sonic G1 encoder(+FSQ)+decoder core."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        enc_group: str = "actor_enc",
        proprio_group: str | None = "actor_proprio",
        enc_dim: int = SonicG1Core.ENC_INPUT_DIM,
        proprio_dim: int = SonicG1Core.PROPRIO_DIM,
        with_kin_aux: bool = False,
        use_fsq: bool = True,
        freeze_encoder: bool = False,
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
        pretrained_ckpt: str | None = None,
    ) -> None:
        del obs_groups, obs_set  # sonic indexes fixed TensorDict keys (HORA-style)
        super().__init__()
        self.enc_group = enc_group
        self.proprio_group = proprio_group
        self.enc_dim = int(enc_dim)
        self.proprio_dim = int(proprio_dim)
        self.obs_dim = self.enc_dim + self.proprio_dim
        self.action_dim = int(output_dim)

        # Forward the obs/action widths into the core so a non-G1 robot (e.g. H2,
        # enc 680 / proprio 990 / action 31) builds a correctly-sized network. Defaults
        # keep G1 (640/930/29) byte-identical.
        self.core = SonicG1Core(
            with_kin_aux=with_kin_aux,
            use_fsq=use_fsq,
            enc_input_dim=self.enc_dim,
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
        )
        if pretrained_ckpt:
            load_g1_from_last_pt(self.core, pretrained_ckpt)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            # Freeze the encoder to keep the FSQ token space fixed (VLA-compat, see
            # sonic_onnx_interface_spec.md red line). FSQ itself has no parameters.
            for p in self.core.encoder.parameters():
                p.requires_grad_(False)

        self.obs_normalizer: nn.Module = (
            EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        )

        dist_cfg = dict(distribution_cfg or {"init_std": 1.0, "std_type": "scalar"})
        dist_cfg.pop("class_name", None)
        self.distribution = GaussianDistribution(self.action_dim, **dist_cfg)

        self._validate_obs(obs)

    # -- obs routing --------------------------------------------------------
    def _validate_obs(self, obs: TensorDict) -> None:
        if self.enc_group not in obs.keys():
            raise KeyError(
                f"SonicG1ActorModel expects obs key '{self.enc_group}'. "
                f"Available: {list(obs.keys())}. Did task-3 env obs extension run?"
            )
        if self.proprio_group is not None and self.proprio_group not in obs.keys():
            raise KeyError(
                f"SonicG1ActorModel expects obs key '{self.proprio_group}'. "
                f"Available: {list(obs.keys())}."
            )

    def _split_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_group is None:
            x = obs[self.enc_group]
            enc = x[..., : self.enc_dim]
            proprio = x[..., self.enc_dim : self.enc_dim + self.proprio_dim]
        else:
            enc = obs[self.enc_group]
            proprio = obs[self.proprio_group]
        return enc, proprio

    def _normalize(self, enc: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.obs_normalizer, nn.Identity):
            return enc, proprio
        x = self.obs_normalizer(torch.cat([enc, proprio], dim=-1))
        return x[..., : self.enc_dim], x[..., self.enc_dim :]

    # -- forward / distribution --------------------------------------------
    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state
        enc, proprio = self._normalize(*self._split_obs(obs))
        mean = self.core.act_mean(enc, proprio)
        self.distribution.update(mean)
        if stochastic_output:
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: Any = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    # -- distribution accessors the PPO runner/algorithm reads -------------
    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return cast("tuple[torch.Tensor, ...]", self.distribution.params)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            enc, proprio = self._split_obs(obs)
            self.obs_normalizer.update(torch.cat([enc, proprio], dim=-1))

    # -- auxiliary recon (g1_kin) for a HIM-PPO-style aux loss (task 5) ----
    def kin_recon(self, obs: TensorDict) -> torch.Tensor | None:
        """Return g1_kin reconstruction of the encoder input, or None if aux disabled.

        Enables sonic's ``g1_recon`` auxiliary loss when a forked PPO algorithm wants
        it: ``MSE(kin_recon(obs), enc_target)``.
        """
        if self.core.kin is None:
            return None
        enc, _ = self._normalize(*self._split_obs(obs))
        return self.core.recon(self.core.encode(enc))

    # -- export ------------------------------------------------------------
    def _export_module(self) -> _SonicInferenceModule:
        return _SonicInferenceModule(
            core=copy.deepcopy(self.core),
            obs_normalizer=copy.deepcopy(self.obs_normalizer),
            enc_dim=self.enc_dim,
            proprio_dim=self.proprio_dim,
        )

    def as_jit(self) -> nn.Module:
        return self._export_module()

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return self._export_module()


class SonicH2ActorModel(SonicG1ActorModel):
    """SONIC actor for the Unitree H2 (31 DOF).

    Identical wiring to ``SonicG1ActorModel`` but with H2's obs/action widths as
    defaults: encoder input 680 = (2·31+6)·10, proprio 990 = (3+3·31+3)·10, so the
    single combined actor stream is 1670 and the decoder emits 31 actions. There is no
    ``last.pt`` in H2 joint layout, so H2 always trains from scratch (leave
    ``pretrained_ckpt`` unset). ``output_dim`` (=31) is supplied by the PPO runner from
    the env action space; ``enc_dim``/``proprio_dim`` may still be overridden in YAML.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("enc_dim", 680)
        kwargs.setdefault("proprio_dim", 990)
        super().__init__(*args, **kwargs)
