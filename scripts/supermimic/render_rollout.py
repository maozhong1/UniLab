"""Render H2 rollouts recorded by record_rollout.py with MuJoCo offscreen rendering.

Solid robot = policy in simulation, translucent blue ghost = reference motion. Writes one mp4 per clip and
optionally a concatenated video. Adapted from the reference SuperMimic scripts/media/render_rollout.py for
the Unilab H2 MJCF (visual meshes in geom group 1, collision geoms in group 0).

Usage:
    MUJOCO_GL=osmesa uv run scripts/supermimic/render_rollout.py --traj media/h2_rollouts.npz \
        --out-dir media/h2_out [--tag "ckpt 45000"] [--concat media/h2_all.mp4]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[2]
H2_XML = ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "h2" / "h2.xml"
ROOT_BODY = "pelvis"
GHOST_RGBA = "0.25 0.50 1.0 0.22"
ROBOT_GROUP, GHOST_GROUP, FLOOR_GROUP = 1, 2, 3


def two_robot_xml(mjcf: Path, root_body: str, width: int, height: int) -> str:
    xml = mjcf.read_text()
    xml = re.sub(
        r'meshdir="([^"]*)"',
        lambda m: f'meshdir="{(mjcf.parent / m.group(1)).resolve()}"',
        xml,
        count=1,
    )
    body = xml[xml.index(f'<body name="{root_body}"') : xml.rindex("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    # ghost: only the visual meshes (contype="0"), drawn translucent in their own group
    ghost = re.sub(
        r"<geom\b[^>]*/>",
        lambda m: m.group(0) if ("mesh=" in m.group(0) and 'contype="0"' in m.group(0)) else "",
        ghost,
    )
    ghost = re.sub(r' (contype|conaffinity|group|material|rgba|class|density)="[^"]*"', "", ghost)
    ghost = ghost.replace(
        "<geom ",
        f'<geom contype="0" conaffinity="0" density="0" group="{GHOST_GROUP}" rgba="{GHOST_RGBA}" ',
    )
    ghost = re.sub(r"<site[^>]*/>", "", ghost)
    scene = f"""
    <light name="top" pos="0 0 5" dir="0 0 -1" directional="true" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0" group="{FLOOR_GROUP}"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.90 0.91 0.93" rgb2="0.76 0.78 0.82" width="512" height="512" mark="edge" markrgb="0.68 0.70 0.74"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0.0" specular="0.05" shininess="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.97 0.97 0.98" rgb2="0.80 0.84 0.90" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    visual = (
        f'<visual><global offwidth="{width}" offheight="{height}"/><quality shadowsize="4096" offsamples="4"/>'
        '<map znear="0.01"/><headlight ambient="0.25 0.25 0.25" diffuse="0.35 0.35 0.35" specular="0.15 0.15 0.15"/></visual>\n'
    )
    xml = re.sub(r"<visual>.*?</visual>", "", xml, flags=re.S)
    xml = xml.replace("<asset>", visual + "  <asset>", 1)
    for tag in ("actuator", "sensor", "keyframe", "contact", "equality"):
        xml = re.sub(rf"<{tag}>.*?</{tag}>", "", xml, flags=re.S)
    return xml


def draw_label(img: np.ndarray, text: str, sub: str, font, font_small) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    pad = 14
    w = max(dr.textlength(text, font=font), dr.textlength(sub, font=font_small)) + 2 * pad
    dr.rounded_rectangle([16, 16, 16 + w, 16 + 2 * pad + 52], radius=10, fill=(20, 24, 32, 150))
    dr.text((16 + pad, 16 + pad - 2), text, font=font, fill=(255, 255, 255, 255))
    dr.text((16 + pad, 16 + pad + 28), sub, font=font_small, fill=(210, 220, 235, 255))
    return np.asarray(im)


def pretty(name: str) -> str:
    base = name.split("__")[0]
    base = re.sub(r"_(R|L|M)$", "", base)
    base = re.sub(r"_\d{3}$", "", base)
    return base.replace("_", " ")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--traj", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument(
        "--seconds", type=float, default=None, help="trim each clip to its first N seconds"
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=140.0)
    p.add_argument("--elevation", type=float, default=-14.0)
    p.add_argument("--distance", type=float, default=3.2)
    p.add_argument("--no-ghost", action="store_true")
    p.add_argument(
        "--tag", type=str, default="", help="prefix of the label sub-line, e.g. 'ckpt 45000'"
    )
    p.add_argument("--clips", nargs="*", default=None)
    p.add_argument("--concat", type=Path, default=None)
    args = p.parse_args()

    d = dict(np.load(args.traj))
    model = mujoco.MjModel.from_xml_string(
        two_robot_xml(H2_XML, ROOT_BODY, args.width, args.height)
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.sitegroup[:] = 0
    for g in (ROBOT_GROUP, GHOST_GROUP, FLOOR_GROUP):
        opt.geomgroup[g] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = font_small = ImageFont.load_default()

    names = [str(n) for n in d["names"]]
    nq_robot = d["robot_qpos"].shape[-1]
    assert model.nq == 2 * nq_robot, (model.nq, nq_robot)
    failed = d.get("failed", np.zeros(len(names), dtype=bool))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(round(50 / args.fps)))
    outputs = []
    for name in [n for n in (args.clips or names) if n in names]:
        i = names.index(name)
        T = int(d["lengths"][i])
        if args.seconds is not None:
            T = min(T, int(args.seconds * 50) + 1)
        qr, qf = d["robot_qpos"][i, :T], d["ref_qpos"][i, :T]
        out = args.out_dir / f"{name}.mp4"
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        target = qr[0, :3].copy()  # smooth camera: follow a low-passed root position
        sub_prefix = f"{args.tag}   " if args.tag else ""
        fail_tag = "   FELL" if failed[i] else ""
        for t in range(0, T, stride):
            data.qpos[:nq_robot] = qr[t]
            data.qpos[nq_robot:] = qf[t]
            if args.no_ghost:
                data.qpos[nq_robot : nq_robot + 3] = [0, 0, -5]
            mujoco.mj_kinematics(model, data)
            target = 0.92 * target + 0.08 * qr[t, :3]
            cam.lookat[:] = [target[0], target[1], 0.75]
            cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation
            renderer.update_scene(data, cam, opt)
            img = draw_label(
                renderer.render(),
                pretty(name),
                f"{sub_prefix}normalized return {float(d['norm_return'][i]):.3f}{fail_tag}   t = {t / 50:4.1f} s",
                font,
                font_small,
            )
            ff.stdin.write(np.ascontiguousarray(img).tobytes())
        ff.stdin.close()
        ff.wait()
        outputs.append(out)
        print("wrote", out, f"{T / 50:.1f} s", flush=True)
    if args.concat and outputs:
        lst = args.out_dir / "concat.txt"
        lst.write_text("".join(f"file '{o.resolve()}'\n" for o in outputs))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", str(args.concat)],
            check=True,
        )  # fmt: skip
        print("wrote", args.concat)


if __name__ == "__main__":
    main()
