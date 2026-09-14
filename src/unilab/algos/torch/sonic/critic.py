"""UniLab RSL-RL critic for SONIC: stock ``MLPModel`` + optional warm-load of the
official critic weights from ``last.pt``.

The official SONIC value net is a plain MLP ``[2048,2048,1024,1024,512,512]`` SiLU with a
front ``RunningMeanStd`` over the 1645-d ``privileged_mf_hist`` observation. Stock
``rsl_rl.models.MLPModel`` already reproduces that architecture index-for-index (Linears at
even Sequential indices 0..12); this subclass only adds a ``pretrained_ckpt`` hook so
``last.pt``'s ``value_state_dict`` can be loaded at construction — mirroring
``SonicG1ActorModel``'s ``pretrained_ckpt`` — which cures the cold critic at the start of
warm training.

Requires the env critic obs to be the official 1645-d layout
(``env.critic_privileged_mf_hist=true``); otherwise the loader raises with an explicit
hint. Keeps ``.mlp`` / ``.obs_normalizer`` (the ``MLPModel`` API) intact —
``FinalObservationAwarePPO._critic_value`` and ``.update_normalization`` depend on them.
"""
from __future__ import annotations

from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .core import load_critic_from_last_pt


class SonicCriticModel(MLPModel):
    """Stock MLP critic that can warm-load the official SONIC critic from ``last.pt``."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        pretrained_ckpt: str | None = None,
        **kwargs,
    ) -> None:
        # hidden_dims / activation / obs_normalization / distribution_cfg flow through kwargs.
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if pretrained_ckpt:
            load_critic_from_last_pt(self, pretrained_ckpt)
