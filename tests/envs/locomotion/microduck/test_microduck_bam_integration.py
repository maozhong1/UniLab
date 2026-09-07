"""Phase 2/3 integration: BAM actuator holds the microduck STAND pose in physics.

Builds a real mujoco_uni backend from the motor model, wires the BAM actuator as
the pre-step control, and drives the STAND keyframe target for ~2 s. Mirrors the
AGENTS.md equilibrium check: hold the target and verify the robot does not
collapse or tumble (base height stays sane AND tilt small), not just height.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.scene import SceneCfg

pytest.importorskip("mujoco")
pytest.importorskip("bam")

from unilab.envs.locomotion.microduck.actuator import BamActuatorConfig, MicroduckBamActuator

N = 4
SIM_DT = 0.005
DECIMATION = 4  # 50 Hz control over 200 Hz physics


def _scene() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "microduck" / "scene_flat_motor.xml")


def test_bam_holds_stand_pose():
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    bkd = MuJoCoBackend(SceneCfg(model_file=_scene()), N, SIM_DT, base_name="trunk_base")
    bkd.materialize()
    assert bkd.model.nu == 14

    # Actuator order defines the ctrl/servo column order.
    servos = list(bkd.get_actuator_names())
    act = MicroduckBamActuator(
        BamActuatorConfig(vin_range=(7.4, 7.4), vin_drop_resistance_range=(0.0, 0.0)),
        N, servos,
    ).bind(bkd)
    bkd.set_pre_step_control(act)

    # Reset every env to the STAND keyframe; hold its ctrl target.
    stand_qpos = bkd.get_keyframe_qpos("STAND")
    qpos = np.tile(stand_qpos, (N, 1))
    bkd.set_state(np.arange(N), qpos, np.zeros((N, bkd.model.nv)))
    target = np.tile(stand_qpos[7:7 + 14], (N, 1)).astype(np.float32)  # 14 servo angles

    z0 = bkd.get_base_pos()[:, 2].copy()
    for _ in range(int(2.0 / (SIM_DT * DECIMATION))):  # ~2 s
        bkd.step(target, nsteps=DECIMATION)

    pos = bkd.get_base_pos()
    quat = bkd.get_base_quat()  # (N,4) wxyz
    assert np.isfinite(pos).all() and np.isfinite(quat).all()

    z = pos[:, 2]
    # Upright: trunk +z tilt from world +z. gravity_z proxy via quat: compute the
    # body-frame world-up projection; tilt = arccos of trunk z-axis · world z.
    w, x, y, zq = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    up_z = 1.0 - 2.0 * (x * x + y * y)  # R[2,2]: trunk local-z projected on world-z
    tilt = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))

    print(f"z0={z0.mean():.3f} z={z.mean():.3f} tilt_max={tilt.max():.1f}deg")
    assert (z > 0.09).all(), f"collapsed: z={z}"      # started ~0.12, didn't sink
    assert (z < 0.16).all(), f"launched: z={z}"
    assert (tilt < 20.0).all(), f"tumbled: tilt={tilt}"
