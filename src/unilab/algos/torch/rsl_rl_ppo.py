from __future__ import annotations

import logging
import math
from typing import Any, cast

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from unilab.algos.torch.common.compile import get_torch_compile_for_cuda

logger = logging.getLogger(__name__)

_LOG_2_PI = math.log(2.0 * math.pi)
_NORMAL_ENTROPY_OFFSET = 0.5 * (1.0 + _LOG_2_PI)


class FinalObservationAwarePPO(PPO):
    """PPO variant that bootstraps time limits from env final_observation."""

    learning_rate: float

    def __init__(
        self,
        *args: Any,
        enable_compile: bool = False,
        encoder_lr: float | None = None,
        critic_lr: float | None = None,
        critic_warmup_iters: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_compile = (
            bool(enable_compile) and get_torch_compile_for_cuda(self.device, warn=True) is not None
        )
        self._minibatch_loss_fn = self._minibatch_loss_tensors
        if self.enable_compile:
            self._compile_training_methods()
        self._setup_lr_groups(encoder_lr, critic_lr)
        self._setup_critic_warmup(critic_warmup_iters)

    def _setup_critic_warmup(self, critic_warmup_iters: int) -> None:
        """Freeze the actor for the first ``critic_warmup_iters`` iterations.

        Root-cause cure for cold-critic warm-start collapse: when a warm (expert) actor
        is paired with a FRESH critic, the critic's early value estimates are garbage, so
        the first PPO advantages are noise that push the expert actor off its good init;
        by the time the critic catches up the actor is already corrupted and the pair
        locks into a low-reward equilibrium. Freezing the actor (encoder + decoder + std)
        for N iters lets the critic burn in on the warm policy's returns FIRST; the actor
        is then unfrozen and updated with advantages from an already-accurate critic.

        Implementation: flip ``requires_grad=False`` on every currently-trainable actor
        tensor (so params already frozen by ``freeze_encoder`` stay frozen and are NOT
        restored later). The critic keeps training normally. Intended to be used together
        with ``critic_lr`` (split optimizer, schedule pinned 'fixed'); with an adaptive
        schedule the near-zero KL during warmup would spuriously nudge the LR.
        """
        self._critic_warmup_iters = int(critic_warmup_iters or 0)
        self._iter_count = 0
        self._warmup_active = False
        self._warmup_frozen_params: list[torch.nn.Parameter] = []
        if self._critic_warmup_iters <= 0:
            return
        self._warmup_frozen_params = [p for p in self.actor.parameters() if p.requires_grad]
        for p in self._warmup_frozen_params:
            p.requires_grad_(False)
        self._warmup_active = True
        if self.schedule == "adaptive":
            logger.warning(
                "[critic_warmup] schedule is 'adaptive'; recommend setting critic_lr so the "
                "schedule is pinned 'fixed' (near-zero KL while the actor is frozen would "
                "otherwise perturb the LR)."
            )
        logger.info(
            f"[critic_warmup] freezing actor for the first {self._critic_warmup_iters} iters "
            f"(cold-critic burn-in); {len(self._warmup_frozen_params)} actor tensors frozen, "
            f"only the critic learns until then."
        )

    def _maybe_end_critic_warmup(self) -> None:
        """Unfreeze the actor once the burn-in window has elapsed (called each update)."""
        if not self._warmup_active or self._iter_count < self._critic_warmup_iters:
            return
        for p in self._warmup_frozen_params:
            p.requires_grad_(True)
        self._warmup_active = False
        self._warmup_frozen_params = []
        logger.info(
            f"[critic_warmup] done after {self._critic_warmup_iters} iters -> actor unfrozen; "
            f"resuming normal actor+critic PPO updates."
        )

    def _setup_lr_groups(self, encoder_lr: float | None, critic_lr: float | None) -> None:
        """Give the encoder and/or the critic their own (absolute) LR param groups.

        No-op when both are None -> the base single param group is kept and encoder,
        decoder and critic all share ``learning_rate`` (default behaviour).

        When either is set, the optimizer is split into up to three groups — encoder,
        critic, and "rest" (decoder + actor std) — each tagged with an ``lr_scale`` =
        group_lr / base_lr. ``base_lr`` (= ``learning_rate``) is the actor/decoder LR.
        This replicates official sonic's fixed separate actor/critic LRs (actor 2e-5,
        critic 1e-3), the standard cure for cold-critic warm-start collapse: the fresh
        critic learns fast while the warm actor barely moves.

        The adaptive-KL scheduler flattens every param group to a single scalar LR each
        minibatch, which would erase the split, so we pin ``schedule='fixed'`` whenever a
        custom group is created. Frozen / non-trainable params are filtered out.
        """
        if encoder_lr is None and critic_lr is None:
            return

        base_lr = float(self.learning_rate)
        # Build the special (non-base) groups first, tracking claimed params so a param
        # is never placed in two groups (e.g. if critic and encoder overlapped).
        special: list[tuple[str, list, float]] = []
        claimed_ids: set[int] = set()

        if critic_lr is not None:
            crit_params = [p for p in self.critic.parameters() if p.requires_grad]
            if crit_params:
                special.append(("critic", crit_params, float(critic_lr)))
                claimed_ids |= {id(p) for p in crit_params}
            else:
                logger.info("[critic_lr] critic has no trainable params; ignoring critic_lr")

        if encoder_lr is not None:
            encoder = getattr(getattr(self.actor, "core", None), "encoder", None)
            if encoder is None:
                logger.warning("[encoder_lr] actor has no .core.encoder; ignoring encoder_lr")
            else:
                enc_params = [
                    p
                    for p in encoder.parameters()
                    if p.requires_grad and id(p) not in claimed_ids
                ]
                if enc_params:
                    special.append(("encoder", enc_params, float(encoder_lr)))
                    claimed_ids |= {id(p) for p in enc_params}
                else:
                    logger.info(
                        "[encoder_lr] encoder has no trainable params (frozen); ignoring encoder_lr"
                    )

        if not special:
            return

        # "rest" = everything currently in the optimizer not claimed by a special group.
        rest_params = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if id(p) not in claimed_ids
        ]
        # Carry over non-lr optimizer hyperparameters (betas, eps, weight_decay, ...).
        template = {
            k: v
            for k, v in self.optimizer.param_groups[0].items()
            if k not in ("params", "lr", "initial_lr", "lr_scale")
        }
        groups = [
            {"params": params, "lr": lr, "lr_scale": lr / base_lr, **template}
            for _, params, lr in special
        ]
        groups.append({"params": rest_params, "lr": base_lr, "lr_scale": 1.0, **template})
        opt_cls = type(self.optimizer)
        self.optimizer = opt_cls(groups)

        if self.schedule == "adaptive":
            logger.warning(
                "[lr_groups] pinning schedule='fixed' (was 'adaptive') so per-group LRs "
                "stay separate; the base (decoder/actor) LR will no longer KL-adapt."
            )
            self.schedule = "fixed"
        desc = ", ".join(f"{n} lr={lr:g}(scale={lr / base_lr:.3g})" for n, _, lr in special)
        logger.info(
            f"[lr_groups] split optimizer -> {desc}, rest(decoder/actor) lr={base_lr:g}"
        )

    def _compile_training_methods(self) -> None:
        compile_fn = get_torch_compile_for_cuda(self.device, warn=True)
        if compile_fn is None:
            return

        self._minibatch_loss_fn = compile_fn(
            self._minibatch_loss_tensors,
            mode="reduce-overhead",
            fullgraph=False,
        )

    @staticmethod
    def _model_obs_tensor(model: Any, obs: TensorDict) -> torch.Tensor:
        obs_groups = getattr(model, "obs_groups", None)
        if not obs_groups:
            raise RuntimeError("PPO compiled update requires model.obs_groups")
        tensors = [obs[group] for group in obs_groups]
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)

    def _supports_compiled_update_path(self) -> bool:
        if not self.enable_compile:
            return False
        if self.rnd or self.symmetry or self.is_multi_gpu:
            return False
        if self.actor.is_recurrent or self.critic.is_recurrent:
            return False
        distribution: Any = getattr(self.actor, "distribution", None)
        if distribution is None or not hasattr(distribution, "std_type"):
            return False
        if distribution.std_type == "scalar":
            return hasattr(distribution, "std_param")
        if distribution.std_type == "log":
            return hasattr(distribution, "log_std_param")
        return False

    def _actor_mean_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution: Any = self.actor.distribution
        if distribution is None:
            raise RuntimeError("PPO actor must expose a stochastic distribution")

        mean = self.actor.mlp(self.actor.obs_normalizer(obs))
        if distribution.std_type == "scalar":
            std = distribution.std_param.expand_as(mean)
        else:
            std = torch.exp(distribution.log_std_param).expand_as(mean)
        return mean, std

    def _critic_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic.mlp(self.critic.obs_normalizer(obs)).squeeze(-1)

    @staticmethod
    def _gaussian_log_prob(
        actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        normalized = (actions - mean) / std
        return (-0.5 * (normalized.pow(2) + 2.0 * torch.log(std) + _LOG_2_PI)).sum(dim=-1)

    @staticmethod
    def _gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
        return (torch.log(std) + _NORMAL_ENTROPY_OFFSET).sum(dim=-1)

    def _minibatch_loss_tensors(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        target_values: torch.Tensor,
        advantages: torch.Tensor,
        old_actions_log_prob: torch.Tensor,
        old_values: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self._actor_mean_std(actor_obs)
        actions_log_prob = self._gaussian_log_prob(actions, mu, sigma)
        values = self._critic_value(critic_obs)
        entropy = self._gaussian_entropy(sigma).mean()

        old_actions_log_prob = old_actions_log_prob.squeeze(-1)
        old_values = old_values.squeeze(-1)
        target_values = target_values.squeeze(-1)
        advantages = advantages.squeeze(-1)

        ratio = torch.exp(actions_log_prob - old_actions_log_prob)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (values - target_values).pow(2)
            value_losses_clipped = (value_clipped - target_values).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (target_values - values).pow(2).mean()

        kl = torch.sum(
            torch.log(sigma / old_sigma + 1e-5)
            + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
            - 0.5,
            dim=-1,
        )
        kl_mean = torch.mean(kl)

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        return loss, surrogate_loss, value_loss, entropy, kl_mean

    def update(self) -> dict[str, float]:
        # Unfreeze the actor once the critic burn-in window elapses (no-op otherwise),
        # then advance the per-iteration counter regardless of which update path runs.
        self._maybe_end_critic_warmup()
        try:
            return self._update_inner()
        finally:
            self._iter_count += 1

    def _update_inner(self) -> dict[str, float]:
        if not self._supports_compiled_update_path():
            return cast(dict[str, float], super().update())

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for batch in generator:
            actions = cast(torch.Tensor, batch.actions)
            values = cast(torch.Tensor, batch.values)
            advantages = cast(torch.Tensor, batch.advantages)
            returns = cast(torch.Tensor, batch.returns)
            old_actions_log_prob = cast(torch.Tensor, batch.old_actions_log_prob)
            old_distribution_params = batch.old_distribution_params
            if old_distribution_params is None:
                raise RuntimeError("PPO compiled update requires old distribution parameters")

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_obs = self._model_obs_tensor(self.actor, batch.observations)
            critic_obs = self._model_obs_tensor(self.critic, batch.observations)
            old_mu, old_sigma = old_distribution_params

            loss, surrogate_loss, value_loss, entropy, kl_mean = self._minibatch_loss_fn(
                actor_obs,
                critic_obs,
                actions,
                returns,
                advantages,
                old_actions_log_prob,
                values,
                old_mu,
                old_sigma,
            )

            if self.desired_kl is not None and self.schedule == "adaptive":
                kl_value = float(kl_mean.detach())
                learning_rate = float(self.learning_rate)
                if kl_value > self.desired_kl * 2.0:
                    learning_rate = max(1e-5, learning_rate / 1.5)
                elif kl_value < self.desired_kl / 2.0 and kl_value > 0.0:
                    learning_rate = min(1e-2, learning_rate * 1.5)

                self.learning_rate = learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = learning_rate * param_group.get("lr_scale", 1.0)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor | TensorDict],
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        timeouts = extras.get("time_outs")
        timeout_bootstrap_obs = extras.get("time_out_bootstrap_obs")
        if isinstance(timeouts, torch.Tensor):
            timeout_mask = timeouts.to(self.device).float()
            if timeout_bootstrap_obs is not None and torch.count_nonzero(timeout_mask) > 0:
                bootstrap_obs = timeout_bootstrap_obs.to(self.device)
                bootstrap_values = self.critic(bootstrap_obs).detach()
                self.transition.rewards += self.gamma * torch.squeeze(
                    bootstrap_values * timeout_mask.unsqueeze(1), 1
                )
            else:
                transition_values = self.transition.values
                assert transition_values is not None
                self.transition.rewards += self.gamma * torch.squeeze(
                    transition_values * timeout_mask.unsqueeze(1), 1
                )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)
