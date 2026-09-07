#!/usr/bin/env python3
"""Generate BAM-ready `<motor>` variants of the microduck MJCF models.

microduck_rl ships position-actuator MJCF (the mjlab/BAM ``edit_spec`` converts
them to motors at load time). UniLab's SceneCfg compiles an XML file directly,
so we materialize the motor variant on disk instead. Mirrors
``bam.mjlab.BamActuator.edit_spec``:

* every servo ``<position class="chosen_actuator" .../>`` -> ``<motor gear=1
  forcerange=±force_limit>`` (torque; ``data.ctrl`` = generalized force, which
  the ``MicroduckBamActuator`` pre-step callback writes each substep);
* the ``chosen_actuator`` joint default gets a constant ``dof_frictionloss`` /
  ``dof_damping`` (nominal BAM gearbox + viscous friction) so MuJoCo's solver
  provides static stiction — the electrical torque is computed live by
  ``MicroduckBamActuator`` but friction MUST live in the solver (see that
  module's docstring). ``armature`` = BAM extra inertia. The friction
  constraint is stiffened (``solreffriction``/``solimpfriction``) exactly like
  training's ``stiff_frictionloss`` to stop a statically-held joint creeping.

Run: ``uv run python scripts/microduck_make_motor_xml.py``
"""

from __future__ import annotations

import re
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.locomotion.microduck.actuator import BamActuatorConfig, MicroduckBamActuator

ROBOTS = ["robot_walk", "robot_groundcontact", "robot_groundcontact_rollers"]
# position-scene -> motor-scene, and the robot include each swaps to.
SCENES = {
    "scene_flat.xml": "robot_walk",
    "scene_groundcontact.xml": "robot_groundcontact",
    "scene_rollers.xml": "robot_groundcontact_rollers",
}

_POS_ACT = re.compile(
    r'<position\s+class="chosen_actuator"\s+name="(?P<name>[^"]+)"\s+joint="(?P<joint>[^"]+)"\s*/>'
)


# Constant dof_frictionloss standing in for BAM's load-dependent gearbox
# friction at a nominal holding torque (~load_friction_motor · 0.2 Nm ≈ 0.05).
# Holds the STAND pose (tilt ~5°) without over-damping the gait. Tunable.
NOMINAL_FRICTIONLOSS = 0.05
# Stiff friction constraint (MuJoCo direct solref form), matching training's
# bam.mjlab stiff_frictionloss — kills static creep.
STIFF_SOLREF = "-5e4 -2e2"
STIFF_SOLIMP = "0.99 0.9999 0.001 0.5 2.0"


def _params() -> tuple[float, float, float]:
    a = MicroduckBamActuator(BamActuatorConfig(), 1, ["left_hip_yaw"])
    return a.armature, a.force_limit, a.friction_viscous


def convert_robot(text: str, armature: float, force_limit: float, viscous: float) -> str:
    # 1. chosen_actuator joint default: constant BAM friction + BAM armature +
    #    stiff friction constraint (solver provides static stiction).
    text, n = re.subn(
        r'(<default class="chosen_actuator">.*?<joint )'
        r'damping="[^"]*" frictionloss="[^"]*" armature="[^"]*"(/>)',
        rf'\g<1>damping="{viscous:.6g}" frictionloss="{NOMINAL_FRICTIONLOSS:.6g}" '
        rf'armature="{armature:.8g}" solreffriction="{STIFF_SOLREF}" '
        rf'solimpfriction="{STIFF_SOLIMP}"\g<2>',
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError("chosen_actuator joint default not found/edited")
    # 2. position actuators -> torque motors.
    fr = f"{-force_limit:.6g} {force_limit:.6g}"
    text, n = _POS_ACT.subn(
        lambda m: f'<motor name="{m["name"]}" joint="{m["joint"]}" gear="1 0 0 0 0 0" '
        f'forcerange="{fr}"/>',
        text,
    )
    if n != 14:
        raise RuntimeError(f"expected 14 position actuators, converted {n}")
    return text


def main() -> None:
    root = Path(ASSETS_ROOT_PATH) / "robots" / "microduck"
    armature, force_limit, viscous = _params()
    print(f"armature={armature:.8g}  force_limit=±{force_limit:.6g}  "
          f"frictionloss={NOMINAL_FRICTIONLOSS} damping={viscous:.6g}")

    for robot in ROBOTS:
        src = (root / f"{robot}.xml").read_text()
        out = convert_robot(src, armature, force_limit, viscous)
        (root / f"{robot}_motor.xml").write_text(out)
        print(f"  wrote {robot}_motor.xml")

    for scene, robot in SCENES.items():
        text = (root / scene).read_text()
        text = text.replace(f'file="{robot}.xml"', f'file="{robot}_motor.xml"')
        motor_scene = scene.replace(".xml", "_motor.xml")
        (root / motor_scene).write_text(text)
        print(f"  wrote {motor_scene}  (includes {robot}_motor.xml)")


if __name__ == "__main__":
    main()
