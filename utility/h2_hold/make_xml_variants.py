"""Write contact-model A/B variants of the H2 MJCF (the repo XML is never modified).

Creates ``<out_root>/xml_{dt2,solref,box}/{h2.xml,scene_flat.xml}`` (+ ``assets`` symlink):

* ``dt2``    -- 2 ms physics timestep (also pass ``--simdt 0.002`` to run_hold.py)
* ``solref`` -- stiffer foot-sphere contact (``solref="0.005 1"``)
* ``box``    -- one flat box sole per foot with the same 4-sphere footprint

usage:
    uv run --no-sync python utility/h2_hold/make_xml_variants.py <out_root>
    # then: run_hold.py ... --scene <out_root>/xml_box/scene_flat.xml
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/unilab/assets/robots/h2"


def emit(out_root: Path, name: str, xml: str, scene: str) -> None:
    d = out_root / f"xml_{name}"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "assets").exists():
        (d / "assets").symlink_to(SRC / "assets")
    (d / "h2.xml").write_text(xml)
    (d / "scene_flat.xml").write_text(scene)
    print(f"wrote {d}/scene_flat.xml")


def box(m: re.Match) -> str:
    # flat box sole: x -0.067..0.147, |y| 0.005..0.07, bottom z=-0.0543
    side = m.group(1)
    sy = 1 if side == "left" else -1
    return (
        f'<geom name="{side}_foot1_collision" type="box" size="0.107 0.0325 0.005" '
        f'pos="0.04 {sy * 0.0375} -0.0493" contype="1" conaffinity="0" condim="3" '
        f'friction="1.0" priority="1" rgba="0.8 0.2 0.2 1" />'
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__)
    out_root = Path(sys.argv[1]).resolve()
    base = (SRC / "h2.xml").read_text()
    scene = (SRC / "scene_flat.xml").read_text()
    if 'timestep="0.005"' not in base or "left_foot1_collision" not in base:
        raise SystemExit("h2.xml layout changed (timestep / foot spheres); update this script")
    emit(out_root, "dt2", base.replace('timestep="0.005"', 'timestep="0.002"'), scene)
    solref = re.sub(
        r'(<geom name="(left|right)_foot\d_collision"[^/]*?)(\s*/>)', r'\1 solref="0.005 1"\3', base
    )
    emit(out_root, "solref", solref, scene)
    x = re.sub(r'<geom name="(left|right)_foot1_collision"[^>]*/>', box, base)
    x = re.sub(r'\s*<geom name="(left|right)_foot[234]_collision"[^>]*/>', "", x)
    emit(out_root, "box", x, scene)


if __name__ == "__main__":
    main()
