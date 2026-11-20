"""World model: learned dynamics + learned termination head + analytic observation / reward heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import AnalyticHeads, ReferenceFrame
from .state_spec import StateSpec, StateSpecCfg


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...],
        out_dim: int,
        act=nn.SiLU,
        zero_last: bool = False,
        last_gain: float | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)
        # Optional reduced-precision matmuls (e.g. torch.bfloat16 on Intel XPU, whose XMX units
        # are ~7x faster than fp32 there); inputs and outputs stay float32.
        self.autocast_dtype: torch.dtype | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.autocast_dtype is None:
            return self.net(x)
        with torch.autocast(device_type=x.device.type, dtype=self.autocast_dtype):
            return self.net(x).float()


def set_mlp_autocast(module: nn.Module, dtype: torch.dtype | None) -> None:
    """Set the matmul autocast dtype of every ``MLP`` inside ``module``."""
    for m in module.modules():
        if isinstance(m, MLP):
            m.autocast_dtype = dtype


class DynamicsMLP(nn.Module):
    """delta = f(canonical(s), a, xi_norm) -> (..., 12+2N); zero-initialised last layer (constant-velocity prior)."""

    def __init__(
        self, feat_dim: int, act_dim: int, xi_dim: int, hidden=(2048, 2048, 2048), out_dim: int = 70
    ):
        super().__init__()
        self.mlp = MLP(feat_dim + act_dim + xi_dim, tuple(hidden), out_dim, zero_last=True)

    def forward(
        self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([feat, action, xi_norm], dim=-1))


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8


class TerminationHead(nn.Module):
    """P(terminal | state, reference) from the analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        nj = self.spec.nj
        self.dynamics = DynamicsMLP(
            self.spec.state_dim, nj, nj + 4, cfg.hidden, self.spec.delta_dim
        )
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        delta = self.dynamics(self.spec.canonical(s), action, self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta)

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # The pelvis is the free-joint body, so its pose is the state's root pose: no FK needed.
        return self.term_head(
            self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)
        )

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat):
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        ref = ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat)
        logits = self.term_head(
            self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)
        )
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, e * e, bce, logits

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = (
            torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False)
            if mode
            else self._loss_step
        )

    def window_loss(self, states, a_exec, xi, done, ref_root_pos, ref_root_quat):
        """k-step autoregressive loss (SuperMimic Eq. 1).

        states (k+1, n, S) logged states; a_exec (k, n, N); xi (n, N+4); done (k, n) termination
        labels of states 1..k; ref_root_pos / ref_root_quat (k, n, ·) reference roots of the k
        transitions (termination features of state j+1).
        """
        k = a_exec.shape[0]
        s = states[0]
        total_state = total_term = 0.0
        acc = 0.0
        err_first = err_last = None
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq, bce, logits = step_fn(
                s, a_exec[j], xi, states[j + 1], done[j], ref_root_pos[j], ref_root_quat[j]
            )
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                err_first = nse if j == 0 else err_first
                err_last = nse
                acc = acc + ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        metrics = {
            "state_loss": (total_state / k).detach(),
            "term_bce": (total_term / k).detach(),
            "nse_1step": err_first.detach(),
            "nse_kstep": err_last.detach(),
            "term_acc": (acc / k).detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def rollout_error(
        self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), batch mean."""
        s = states[0]
        out = []
        for j in range(a_exec.shape[0]):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
