"""BAM XL330 actuator adapter for UniLab's MuJoCo (mujoco_uni) backend.

microduck policies are trained against the BAM M6 voltage/friction model
(``bam.mjlab.BamActuator`` in MuJoCo-Warp), not an ideal PD — at this scale the
actuator IS most of the sim2real gap. This module reproduces the *electrical*
half of that model for UniLab's CPU backend as a ``set_pre_step_control``
callback: every physics substep it reads the servo joint state and runs BAM's
firmware voltage control law + DC-motor back-EMF torque equation, vectorized
over ``(num_envs, 14)`` in numpy. The result is written to the ``<motor>``
actuators' ``data.ctrl`` (generalized force).

Friction split — solver-side, not folded into torque
-----------------------------------------------------
BAM's friction budget (Coulomb + viscous + load-dependent gearbox) is what
provides *static holding* (stiction) on a ~0.8 kg biped standing on low-Kp
servos. In training that budget is written into MuJoCo's per-DOF
``dof_frictionloss`` each step and the constraint solver clips it (BAM
Algorithm 1). We tried folding friction into the returned torque here; it CANNOT
hold statically because the callback never sees the gravity/contact load, so the
robot creeps and topples. UniLab's ``mujoco_uni`` batched pool does not expose
per-world ``dof_frictionloss``, so instead the friction is baked into the motor
MJCF as a constant ``dof_frictionloss`` / ``dof_damping`` (see
``scripts/microduck_make_motor_xml.py``) and MuJoCo's solver does the stiction.
The constant is a nominal-load approximation of BAM's load-dependent gearbox
friction (≈ ``load_friction_motor·|τ|`` at typical holding torques ~0.2 Nm).

v1 fidelity gaps (documented; revisit if sim2real transfer needs them):
* friction is constant, not per-step load-dependent, and not per-env — so
  per-env friction DR and the Stribeck low-speed regime are dropped;
* only the electrical dynamics are per-env randomized (vin, back-EMF, Kp).
UniLab retrains from scratch against THIS model, so train/eval stay
self-consistent; Phase 6 cross-checks the exported ONNX in microduck_rl's
infer_policy.py via the shared 61D obs contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# NOTE: ``bam`` (better-actuator-models, the ``microduck`` extra) is imported
# lazily inside ``MicroduckBamActuator.__init__`` so the microduck modules —
# and thus the task registry bootstrap — import cleanly even when the extra is
# absent. Only *constructing* an actuator (i.e. building an env) needs ``bam``.


@dataclass
class BamActuatorConfig:
    """Config for the microduck BAM adapter (mirrors microduck_rl _BAM_ACTUATOR_KWARGS)."""

    motor_name: str = "xl330"
    model: str = "m6"
    kp_fw: float = 200.0            # microduck's preserved firmware stiffness
    # Per-env battery-voltage DR (held constant across resets, sampled at build).
    vin_range: tuple[float, float] = (6.5, 8.2)
    # Load-dependent sag V_drop = R_drop * Σ|τ|/kt, R_drop sampled per-env.
    vin_drop_resistance_range: tuple[float, float] = (0.0, 0.2)
    vin_min: float = 6.0            # floor on effective voltage after sag
    max_current: float | None = None  # training runs WITHOUT the firmware limiter
    # DR ranges consumed by the env's DR provider (Phase 3); defaults = nominal.
    kp_scale_range: tuple[float, float] = (1.0, 1.0)
    kd_scale_range: tuple[float, float] = (1.0, 1.0)


class MicroduckBamActuator:
    """Vectorized BAM voltage actuator, usable as a pre_step_control callback.

    Wire it with ``backend.set_pre_step_control(actuator.bind(backend))``.
    ``__call__(backend, ctrl)`` receives the owner-level control (a position
    target = ``action*scale + default_angles``, shape ``(N, 14)``) and returns
    motor torques of the same shape for the ``<motor>`` actuators. Friction is
    handled by the MJCF (see module docstring), NOT here.
    """

    def __init__(self, cfg: BamActuatorConfig, num_envs: int, servo_names: list[str]):
        self.cfg = cfg
        self.num_envs = num_envs
        self.servo_names = list(servo_names)
        self.n = len(self.servo_names)

        from bam.model import load_model

        m = load_model(motor_name=cfg.motor_name, model=cfg.model)
        act = m.actuator
        act.kp = cfg.kp_fw
        act.max_current = cfg.max_current if (cfg.max_current and cfg.max_current > 0) else None

        # Scalar motor constants (numpy fast path — no torch/backend object mutation).
        self.kt = float(m.kt.value)
        self.R = float(m.R.value)
        self.kp_fw = float(cfg.kp_fw)
        self.error_gain = float(getattr(act, "error_gain", 1.0))
        self.max_pwm = float(getattr(act, "max_pwm", 1.0))
        self.armature = float(act.get_extra_inertia())
        self.friction_base = float(m.friction_base.value)
        self.friction_viscous = float(m.friction_viscous.value)
        # Nominal load-dependent gearbox friction coefficient (motor side), used
        # by the XML generator to pick the constant dof_frictionloss.
        self.load_friction_motor = float(
            getattr(getattr(m, "load_friction_motor", None), "value", 0.0)
        )

        # Upper-bound force limit (safe ceiling regardless of per-env voltage).
        self.force_limit = max(cfg.vin_range) * self.kt / self.R

        # Per-env DR state (numpy, shape (N, 1) for broadcast over joints).
        rng = np.random.default_rng()
        self.vin = _sample(rng, cfg.vin_range, num_envs)
        self.vin_drop_resistance = _sample(rng, cfg.vin_drop_resistance_range, num_envs)
        self.kp_scale = _sample(rng, cfg.kp_scale_range, num_envs)
        self.kd_scale = _sample(rng, cfg.kd_scale_range, num_envs)
        self._default = dict(
            vin=self.vin.copy(), vin_drop_resistance=self.vin_drop_resistance.copy(),
            kp_scale=self.kp_scale.copy(), kd_scale=self.kd_scale.copy(),
        )

        self._pos_idx: np.ndarray | None = None
        self._vel_idx: np.ndarray | None = None
        # Previous applied torque — only used as the battery-sag current proxy.
        self._prev_torque = np.zeros((num_envs, self.n), dtype=np.float64)

    # ── wiring ────────────────────────────────────────────────────────────────
    def bind(self, backend: Any) -> "MicroduckBamActuator":
        """Resolve servo dof indices from a materialized backend. Returns self."""
        self._pos_idx = np.asarray(backend.get_joint_dof_pos_indices(self.servo_names))
        self._vel_idx = np.asarray(backend.get_joint_dof_vel_indices(self.servo_names))
        if self._pos_idx.shape[0] != self.n or self._vel_idx.shape[0] != self.n:
            raise ValueError(
                f"BAM adapter: resolved {self._pos_idx.shape[0]} pos / "
                f"{self._vel_idx.shape[0]} vel indices for {self.n} servos"
            )
        return self

    # ── DR hooks (called by the env's DR provider at reset) ────────────────────
    def set_gains(self, env_ids, kp_scale=None, kd_scale=None) -> None:
        if kp_scale is not None:
            self.kp_scale[env_ids] = np.reshape(kp_scale, (-1, 1))
        if kd_scale is not None:
            self.kd_scale[env_ids] = np.reshape(kd_scale, (-1, 1))

    def reset(self, env_ids) -> None:
        """Restore per-reset DR (kp/kd) to nominal; vin/R are startup-held."""
        self.kp_scale[env_ids] = self._default["kp_scale"][env_ids]
        self.kd_scale[env_ids] = self._default["kd_scale"][env_ids]
        self._prev_torque[env_ids] = 0.0

    # ── the pre_step_control callback ──────────────────────────────────────────
    def __call__(self, backend: Any, ctrl: np.ndarray) -> np.ndarray:
        assert self._pos_idx is not None, "call bind(backend) before stepping"
        q_target = np.asarray(ctrl, dtype=np.float64)                    # (N, 14)
        q = np.asarray(backend.get_dof_pos())[:, self._pos_idx]          # (N, 14)
        dq = np.asarray(backend.get_dof_vel())[:, self._vel_idx]         # (N, 14)

        # Per-env supply voltage with load-dependent sag (I ≈ Σ|τ_prev|/kt).
        vin = self.vin
        if np.any(self.vin_drop_resistance > 0.0):
            current = np.sum(np.abs(self._prev_torque), axis=1, keepdims=True) / self.kt
            vin = np.maximum(vin - self.vin_drop_resistance * current, self.cfg.vin_min)

        # 1. Firmware voltage control law: duty = (target-q)*kp*egain, clamp, *vin.
        duty = (q_target - q) * (self.kp_fw * self.kp_scale) * self.error_gain
        if self.cfg.max_current is not None:
            back_emf = self.kt * dq
            span = self.R * self.cfg.max_current / vin
            center = back_emf / vin
            duty = np.clip(duty, center - span, center + span)
        duty = np.clip(duty, -self.max_pwm, self.max_pwm)
        volts = vin * duty                                              # (N, 14)

        # 2. DC-motor torque with back-EMF (kd_scale scales the electrical damping).
        tau = self.kt * volts / self.R - (self.kt**2) * (dq * self.kd_scale) / self.R
        np.clip(tau, -self.force_limit, self.force_limit, out=tau)
        self._prev_torque = tau
        return tau.astype(ctrl.dtype, copy=False)


def _sample(rng: np.random.Generator, rng_tuple: tuple[float, float], n: int) -> np.ndarray:
    lo, hi = rng_tuple
    if lo == hi:
        return np.full((n, 1), lo, dtype=np.float64)
    return rng.uniform(lo, hi, size=(n, 1)).astype(np.float64)
