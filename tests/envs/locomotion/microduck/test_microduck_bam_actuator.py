"""Phase 2 unit test: the BAM actuator adapter's control law is sane.

Drives MicroduckBamActuator through a tiny fake backend (no physics) to check:
vectorization over (N, 14), finiteness, motor constants match the trained model,
the voltage law pushes toward the target with the right sign, back-EMF opposes
motion, and the DR hooks (kp/friction scale) actually move the output.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bam", reason="better-actuator-models (microduck extra) not installed")

from unilab.envs.locomotion.microduck.actuator import BamActuatorConfig, MicroduckBamActuator

SERVOS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
N = 4


class _FakeBackend:
    """Minimal backend: 14 servos at dof indices 6..19 (after a 6-dof free joint)."""

    sim_dt = 0.005

    def __init__(self, q: np.ndarray, dq: np.ndarray):
        self._q, self._dq = q, dq  # (N, 20) full dof arrays
        self._idx = np.arange(6, 20)

    def get_joint_dof_pos_indices(self, names):
        assert list(names) == SERVOS
        return self._idx

    def get_joint_dof_vel_indices(self, names):
        return self._idx

    def get_dof_pos(self):
        return self._q

    def get_dof_vel(self):
        return self._dq


def _backend(q_servo: np.ndarray, dq_servo: np.ndarray) -> _FakeBackend:
    q = np.zeros((N, 20)); dq = np.zeros((N, 20))
    q[:, 6:20] = q_servo; dq[:, 6:20] = dq_servo
    return _FakeBackend(q, dq)


def _act(**cfg_kw) -> MicroduckBamActuator:
    return MicroduckBamActuator(BamActuatorConfig(**cfg_kw), N, SERVOS)


def test_motor_constants_match_trained_model():
    a = _act(vin_range=(7.4, 7.4))
    assert a.kt == pytest.approx(0.3660, abs=1e-3)
    assert a.R == pytest.approx(2.8114, abs=1e-3)
    assert a.n == 14 and a.force_limit > 0


def test_callback_shape_and_finite():
    a = _act()
    bkd = _backend(np.zeros((N, 14)), np.zeros((N, 14)))
    a.bind(bkd)
    ctrl = np.zeros((N, 14), dtype=np.float32)
    tau = a(bkd, ctrl)
    assert tau.shape == (N, 14) and tau.dtype == np.float32
    assert np.isfinite(tau).all()
    assert np.all(np.abs(tau) <= a.force_limit + 1e-6)


def test_voltage_law_sign_pushes_to_target():
    a = _act(vin_range=(7.4, 7.4), vin_drop_resistance_range=(0.0, 0.0))
    q = np.zeros((N, 14)); dq = np.zeros((N, 14))
    bkd = _backend(q, dq); a.bind(bkd)
    # Positive position error → positive torque (drive toward target).
    tau_pos = a(bkd, np.full((N, 14), 0.3, dtype=np.float32))
    a2 = _act(vin_range=(7.4, 7.4), vin_drop_resistance_range=(0.0, 0.0)); a2.bind(bkd)
    tau_neg = a2(bkd, np.full((N, 14), -0.3, dtype=np.float32))
    assert np.all(tau_pos > 0) and np.all(tau_neg < 0)


def test_back_emf_opposes_motion():
    # At the target (zero pos error) a positive joint velocity must yield a
    # negative (braking) torque from back-EMF + friction.
    a = _act(vin_range=(7.4, 7.4), vin_drop_resistance_range=(0.0, 0.0))
    bkd = _backend(np.zeros((N, 14)), np.full((N, 14), 2.0))
    a.bind(bkd)
    tau = a(bkd, np.zeros((N, 14), dtype=np.float32))
    assert np.all(tau < 0)


def test_dr_hooks_move_output():
    q = np.zeros((N, 14)); dq = np.zeros((N, 14))
    bkd = _backend(q, dq)
    base = _act(vin_range=(7.4, 7.4), kp_scale_range=(1.0, 1.0)); base.bind(bkd)
    t1 = base(bkd, np.full((N, 14), 0.2, dtype=np.float32)).copy()
    # Halving kp_scale must reduce the drive torque at fixed error.
    base.set_gains(np.arange(N), kp_scale=np.full(N, 0.5))
    t2 = base(bkd, np.full((N, 14), 0.2, dtype=np.float32))
    assert np.all(t2 < t1)
