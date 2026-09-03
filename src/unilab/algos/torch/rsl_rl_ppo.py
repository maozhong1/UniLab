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
        target_kl_stop: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # The schedule the user configured (before _setup_lr_groups may pin it 'fixed' to
        # protect split LRs, and before critic_warmup temporarily pins it). critic_warmup
        # restores THIS on unfreeze, so "warmup ends -> back to configured schedule".
        self._configured_schedule = str(self.schedule)
        self.enable_compile = (
            bool(enable_compile) and get_torch_compile_for_cuda(self.device, warn=True) is not None
        )
        self._minibatch_loss_fn = self._minibatch_loss_tensors
        if self.enable_compile:
            self._compile_training_methods()
        self._setup_lr_groups(encoder_lr, critic_lr)
        self._setup_critic_warmup(critic_warmup_iters)
        # Per-update KL early-stop guardrail. When set, the update loop stops applying
        # gradient steps for the current iteration once a minibatch's mean KL exceeds this
        # threshold. Unlike the adaptive-KL *schedule* (which is disabled when critic_lr
        # pins schedule='fixed'), this brake works with a fixed/split-LR optimizer and on
        # the eager (non-compiled) path, so it protects out-of-distribution warm finetunes
        # from drifting off the imported policy. None = disabled.
        self._target_kl_stop = float(target_kl_stop) if target_kl_stop is not None else None
        self._warned_kl_stop_fallback = False
        if self._target_kl_stop is not None:
            logger.info(
                f"[target_kl_stop] KL early-stop guardrail active: stop applying gradient "
                f"steps for an iteration once minibatch mean KL > {self._target_kl_stop:g}."
            )

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
        restored later). The critic keeps training normally.

        Adaptive-schedule safety: while the actor is frozen the KL between old/new policy is
        ~0, which an ``adaptive`` schedule would misread as "too small" and keep inflating
        the LR (ballooning the critic's LR through the burn-in). So if the schedule is
        adaptive we temporarily pin it to 'fixed' for the warmup window and restore
        'adaptive' on unfreeze. (If ``critic_lr`` already pinned 'fixed', nothing to do.)
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
        # Always run the burn-in window under a 'fixed' schedule: the actor is frozen so
        # KL ~ 0, which an adaptive schedule would misread as "too small" and keep inflating
        # the LR (ballooning the critic's LR through burn-in). The configured schedule is
        # restored on unfreeze (see _maybe_end_critic_warmup).
        if self.schedule != "fixed":
            logger.info(
                "[critic_warmup] pinning schedule='fixed' for the burn-in window (frozen actor "
                "=> KL~0 would otherwise inflate the LR)."
            )
            self.schedule = "fixed"
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
        # Restore the configured schedule on unfreeze -- BUT ONLY when there is no split LR.
        #
        # Why the guard: the adaptive-KL scheduler scales the SHARED scalar ``self.learning_rate``
        # and then multiplies every param group by its ``lr_scale``. With a split critic LR the
        # critic's lr_scale is large (e.g. critic_lr=2e-4 over base 5e-6 => scale=40). When the
        # warm policy is near-optimal its KL sits below desired_kl/2, so adaptive keeps INFLATING
        # the base LR every iter; multiplied by 40 the critic LR runs away (base clamp ceiling 1e-2
        # => critic LR up to 0.4), the value function diverges (value_loss 0.3 -> 1e18), advantages
        # turn to garbage and the policy collapses a few iters later. Empirically confirmed on the
        # 2026-09-03 run (value_loss explodes iter34-40, reward/eplen cliff at iter41). So: adaptive
        # and a split LR must NOT coexist. Keep 'fixed' after warmup whenever a split is active;
        # only restore adaptive for the single-LR case.
        note = ""
        if getattr(self, "_has_split_lr", False):
            note = " (schedule kept 'fixed': a split LR is active; adaptive would run the critic LR away)"
        elif self.schedule != self._configured_schedule:
            self.schedule = self._configured_schedule
            note = f" (schedule restored to '{self._configured_schedule}')"
        logger.info(
            f"[critic_warmup] done after {self._critic_warmup_iters} iters -> actor unfrozen; "
            f"resuming normal actor+critic PPO updates.{note}"
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

        Base rsl-rl's adaptive-KL scheduler flattens every param group to a single scalar LR
        each minibatch, which would erase the split, so we pin ``schedule='fixed'`` whenever
        a custom group is created. (The in-class eager loops instead scale by ``lr_scale``,
        so when a split is active AND the schedule is adaptive, ``update`` routes through the
        eager loop to keep the split intact.) Frozen / non-trainable params are filtered out.
        """
        self._has_split_lr = False
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
        self._has_split_lr = True

        if self.schedule == "adaptive":
            # Pin fixed so a split LR stays split, and KEEP it fixed for the whole run --
            # adaptive scales the shared base LR, which the critic's large lr_scale amplifies
            # into a runaway critic LR (value divergence). A critic_warmup does NOT restore
            # adaptive while a split is active; it only restores adaptive in the single-LR case.
            logger.info(
                "[lr_groups] pinning schedule='fixed' (was 'adaptive') so per-group LRs stay "
                "separate; stays fixed after any critic_warmup (adaptive + split LR diverges)."
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

    def _eager_loop_eligible(self) -> bool:
        """Whether the in-class minibatch loop can run (same math as base rsl-rl).

        Independent of ``enable_compile``: the eager loop is also used to honor
        ``target_kl_stop`` on the non-compiled (e.g. XPU) path. Excludes the features the
        in-class loop does not reimplement (RND, symmetry, multi-GPU, recurrent policies).
        """
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

    def _supports_compiled_update_path(self) -> bool:
        if not self.enable_compile:
            return False
        return self._eager_loop_eligible()

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
            # The eager base-mirroring loop (model-agnostic rsl-rl API; works with sonic's
            # non-MLP actor) is used on the non-compiled path when EITHER a KL early-stop is
            # requested OR a split LR must coexist with an adaptive schedule (base rsl-rl
            # would flatten the split; the eager loop preserves it via lr_scale). Otherwise
            # defer to _update_inner -> base update. When neither the compiled nor the eager
            # loop can serve a requested feature, warn once.
            if not self._supports_compiled_update_path():
                needs_eager = self._target_kl_stop is not None or (
                    self.schedule == "adaptive" and getattr(self, "_has_split_lr", False)
                )
                if needs_eager:
                    if self._eager_loop_eligible():
                        return self._update_eager()
                    if self._target_kl_stop is not None and not self._warned_kl_stop_fallback:
                        logger.warning(
                            "[target_kl_stop] set but the update loop cannot be mirrored "
                            "(rnd/symmetry/multi-gpu/recurrent policy); KL early-stop is INACTIVE."
                        )
                        self._warned_kl_stop_fallback = True
            return self._update_inner()
        finally:
            self._iter_count += 1

    def _update_eager(self) -> dict[str, float]:
        """Faithful mirror of base rsl-rl ``PPO.update`` on the eager (non-compiled) path.

        Uses only the model-agnostic actor/critic API (``actor(obs, ...)``,
        ``get_output_log_prob``, ``get_kl_divergence``, ``output_distribution_params``,
        ``output_entropy``), so it works with sonic's encoder+decoder actor — unlike the
        MLP-specialized compiled loop. Excludes RND/symmetry/multi-GPU/recurrent (gated out
        by ``_eager_loop_eligible``).

        Adds two things over base rsl-rl: (1) if ``target_kl_stop`` is set, once a minibatch's
        mean KL exceeds it the loop stops applying steps for this iteration (the offending
        step is skipped), capping per-iter actor movement under a fixed/split LR; (2) the
        adaptive-KL LR update scales each param group by its ``lr_scale``, so an adaptive
        schedule preserves a split LR instead of flattening it.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        n_applied = 0
        kl_stopped = False

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            # Recompute log-prob/entropy/value under the current params.
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            distribution_params = tuple(p for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            # KL divergence for the guardrail (and, when adaptive, the LR schedule).
            with torch.inference_mode():
                kl = self.actor.get_kl_divergence(
                    batch.old_distribution_params, distribution_params
                )
                kl_mean = torch.mean(kl)

            # KL early-stop: skip this (over-shooting) step and stop optimizing this iter.
            if self._target_kl_stop is not None and float(kl_mean) > self._target_kl_stop:
                kl_stopped = True
                break

            if self.desired_kl is not None and self.schedule == "adaptive":
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate * param_group.get("lr_scale", 1.0)

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            n_applied += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        if kl_stopped and (n_applied == 0 or self._iter_count % 10 == 0):
            logger.info(
                f"[target_kl_stop] iter {self._iter_count}: stopped after "
                f"{n_applied}/{num_updates} minibatch updates (mean KL > "
                f"{self._target_kl_stop:g})."
            )
        denom = max(1, n_applied)
        self.storage.clear()
        return {
            "value": mean_value_loss / denom,
            "surrogate": mean_surrogate_loss / denom,
            "entropy": mean_entropy / denom,
        }

    def _update_inner(self) -> dict[str, float]:
        # The in-class (MLP-specialized) loop only runs on the compiled path. Sonic's
        # custom actor is not MLP-shaped, so for it this path is never taken (enable_compile
        # is false); the eager path (target_kl_stop and/or split-LR+adaptive) is handled in
        # update() via _update_eager, which uses the model-agnostic rsl-rl API.
        if not self._supports_compiled_update_path():
            return cast(dict[str, float], super().update())

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        n_applied = 0
        kl_stopped = False

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

            # KL early-stop guardrail: once a minibatch's mean KL exceeds the threshold,
            # skip this (over-shooting) step and stop optimizing for the rest of the
            # iteration. Checked before the step so the offending update is never applied.
            if self._target_kl_stop is not None and float(kl_mean.detach()) > self._target_kl_stop:
                kl_stopped = True
                break

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

            n_applied += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        # Log when the brake engages: always on a full stop (nothing applied), else throttled.
        if kl_stopped and (n_applied == 0 or self._iter_count % 10 == 0):
            logger.info(
                f"[target_kl_stop] iter {self._iter_count}: stopped after "
                f"{n_applied}/{num_updates} minibatch updates (mean KL > "
                f"{self._target_kl_stop:g})."
            )
        denom = max(1, n_applied)
        self.storage.clear()
        return {
            "value": mean_value_loss / denom,
            "surrogate": mean_surrogate_loss / denom,
            "entropy": mean_entropy / denom,
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
