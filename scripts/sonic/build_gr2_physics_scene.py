"""Generate the lightweight GR2 SONIC physics scene used for domain-randomized training.

MuJoCo pools clone the whole mjModel for every env whose model fields are patched at
reset, and the full GR2 scene is ~50 MB (high-poly STL BVHs + skybox texture). This
script writes ``scene_sonic_27dof_physics.xml`` with:

* visual-only mesh geoms (contype=0, conaffinity=0) and textures removed;
* every collision mesh replaced inline by its convex hull capped at ``--maxhullvert``
  vertices (MuJoCo collides meshes through their convex hull anyway).

Bodies, inertials, joints, actuators, sensors, the foot-sole boxes and the keyframe are
unchanged. Rendering keeps using ``scene_sonic_27dof.xml`` via ``visual_model_file``.

    uv run --no-sync python scripts/sonic/build_gr2_physics_scene.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

_GR2_DIR = Path(__file__).resolve().parents[2] / "src" / "unilab" / "assets" / "robots" / "gr2"
_SOURCE = _GR2_DIR / "scene_sonic_27dof.xml"
_OUTPUT = _GR2_DIR / "scene_sonic_27dof_physics.xml"


def _stripped_spec(source: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(source))
    for geom in list(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.contype == 0 and geom.conaffinity == 0:
            spec.delete(geom)
    used = {geom.meshname for geom in spec.geoms if geom.type == mujoco.mjtGeom.mjGEOM_MESH}
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            spec.delete(mesh)
    for material in spec.materials:
        material.textures = [""] * len(material.textures)
    for texture in list(spec.textures):
        spec.delete(texture)
    return spec


def _convex_hull(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return hull vertices (original asset frame) and faces from the compiled mesh graph."""
    graph = model.mesh_graph[model.mesh_graphadr[mesh_id] :]
    num_vert, num_face = int(graph[0]), int(graph[1])
    face_offset = 2 + 3 * num_vert + 3 * num_face
    faces = graph[face_offset : face_offset + 3 * num_face].reshape(num_face, 3)
    vert_start = model.mesh_vertadr[mesh_id]
    verts = model.mesh_vert[vert_start : vert_start + model.mesh_vertnum[mesh_id]]
    used = np.unique(faces)
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.mesh_quat[mesh_id])
    hull_verts = verts[used].astype(np.float64) @ rot.reshape(3, 3).T + model.mesh_pos[mesh_id]
    return hull_verts, remap[faces]


def build(source: Path, maxhullvert: int) -> str:
    hull_spec = _stripped_spec(source)
    for mesh in hull_spec.meshes:
        mesh.maxhullvert = maxhullvert
    hull_model = hull_spec.compile()

    spec = _stripped_spec(source)
    for mesh in spec.meshes:
        mesh_id = mujoco.mj_name2id(hull_model, mujoco.mjtObj.mjOBJ_MESH, mesh.name)
        verts, faces = _convex_hull(hull_model, mesh_id)
        mesh.file = ""
        mesh.uservert = verts.reshape(-1).tolist()
        mesh.userface = faces.reshape(-1).tolist()
    spec.compile()
    header = (
        f"<!-- Generated from {source.name} by scripts/sonic/build_gr2_physics_scene.py "
        f"(maxhullvert={maxhullvert}); do not edit by hand. -->\n"
    )
    return header + spec.to_xml()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maxhullvert", type=int, default=128)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(build(_SOURCE, args.maxhullvert))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
