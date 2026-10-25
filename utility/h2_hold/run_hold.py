"""Hold-at-clip-end rollout for an H2 SONIC policy (CPU, 1 env, deterministic actor).

Builds the env exactly like the ``scripts/train_rsl_rl.py`` play path (Hydra compose of
``task=h2_motion_tracking/sonic_full_train``) with early termination, observation noise and
domain randomisation disabled, plays a ``*_hold.npz`` clip (see ``make_hold_clips.py``) from
frame 0 and records per-step torques, PD error, CoM / CoP / contact data into an npz that
``analyze.py`` / ``compare.py`` / ``trace.py`` consume.

By default the reset is unperturbed. ``--pose-noise`` / ``--vel-noise`` (with ``--seed``) set
small symmetric ``env.pose_randomization`` / ``env.velocity_randomization`` ranges so several
perturbed rollouts per clip can be compared.

usage:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 uv run --no-sync python utility/h2_hold/run_hold.py \
        <clip_hold.npz> <out.npz> --ckpt <model_XXXX.pt> \
        [--n-orig N] [--scene <dir>/scene_flat.xml] [--simdt 0.002] \
        [--seed 1 --pose-noise 0.02 --vel-noise 0.05]

``--n-orig`` (original clip length) defaults to the entry in ``manifest.json`` next to the clip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("EXPORT_POLICY", "0")

import mujoco
import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "scripts", ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

BIG = 1e3
AXES = ("x", "y", "z", "roll", "pitch", "yaw")
KEYS = (
    "frame", "tau", "pderr", "act", "q", "qpos", "com_h", "toe_m", "F", "cop_loc", "cop_lat",
    "cop_h", "gF", "gpen", "gv", "quad", "sole", "rp", "rp_ref", "pel_xy", "pel_ref_xy", "pel_z",
    "foot_xy", "done", "trip",
)  # fmt: skip


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("clip", type=Path, help="*_hold.npz clip")
    ap.add_argument("out", type=Path, help="output npz")
    ap.add_argument("--ckpt", required=True, help="policy checkpoint (model_XXXX.pt)")
    ap.add_argument("--n-orig", type=int, default=None, help="original clip frames (hold start)")
    ap.add_argument("--scene", default=None, help="override env.scene.model_file")
    ap.add_argument("--simdt", type=float, default=None, help="override env.sim_dt")
    ap.add_argument("--seed", type=int, default=0, help="numpy/torch seed (reset sampling)")
    ap.add_argument(
        "--pose-noise",
        type=float,
        default=0.0,
        help="reset pose noise: x/y/roll/pitch/yaw ~ U(-a, a) (m / rad), z ~ U(0, a)",
    )
    ap.add_argument(
        "--vel-noise",
        type=float,
        default=0.0,
        help="reset root velocity noise: lin/ang ~ U(-a, a) (m/s / rad/s)",
    )
    return ap.parse_args()


def resolve_n_orig(clip: Path, n_orig: int | None) -> int:
    if n_orig is not None:
        return n_orig
    manifest = clip.parent / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"--n-orig not given and {manifest} missing (run make_hold_clips.py)")
    table = json.loads(manifest.read_text())
    if clip.name not in table:
        raise SystemExit(f"{clip.name} not in {manifest}")
    return int(table[clip.name])


def build_overrides(args: argparse.Namespace) -> list[str]:
    pn, vn = args.pose_noise, args.vel_noise
    pose = {k: (0.0, pn) if k == "z" else (-pn or 0.0, pn) for k in AXES}
    vel = {k: (-vn or 0.0, vn) for k in AXES}
    overrides = [
        "task=h2_motion_tracking/sonic_full_train",
        "training.device=cpu",
        "training.play_only=true",
        "training.play_env_num=1",
        "env.critic_privileged_mf_hist=true",
        f"+env.motion_file=[{args.clip.resolve()}]",
        "++env.sampling_mode=start",
        "++env.max_episode_seconds=1000",
        # no early termination (training-threshold trips are recorded as `trip` instead)
        f"++env.anchor_pos_z_threshold={BIG}",
        f"++env.anchor_ori_threshold={BIG}",
        f"++env.ee_body_pos_z_threshold={BIG}",
        f"++env.strict_height_threshold={BIG}",
        f"++env.strict_anchor_ori_error_sq={BIG}",
        f"++env.strict_foot_pos_threshold={BIG}",
        f"++env.low_reference_height_threshold={BIG}",
        # reset pose/vel randomisation only via --pose-noise/--vel-noise; no obs noise
        # (all domain_rand flags default off)
        *[f"++env.pose_randomization.{k}=[{lo},{hi}]" for k, (lo, hi) in pose.items()],
        *[f"++env.velocity_randomization.{k}=[{lo},{hi}]" for k, (lo, hi) in vel.items()],
        "++env.noise_config.level=0.0",
        "++env.joint_position_range=[0.0,0.0]",
    ]
    if args.scene:
        overrides.append(f"++env.scene.model_file={args.scene}")
    if args.simdt:
        overrides.append(f"++env.sim_dt={args.simdt}")
    return overrides


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rp_of(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def heading_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])


class FootModel:
    """Foot contact geometry (cold path: built once from the compiled MjModel)."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.fb = [mujoco.mj_name2id(m, 1, f"{s}_ankle_pitch_link") for s in ("left", "right")]
        self.fgeom = [
            [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])
            ]
            for b in self.fb
        ]
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.toe_x = [max(self._extent(g)[1] for g in gs) for gs in self.fgeom]
        self.heel_x = [min(self._extent(g)[0] for g in gs) for gs in self.fgeom]
        self.ctr = [
            np.array(
                [
                    (self.toe_x[k] + self.heel_x[k]) / 2,
                    np.mean([m.geom_pos[g][1] for g in gs]),
                    np.mean([m.geom_pos[g][2] for g in gs]),
                ]
            )
            for k, gs in enumerate(self.fgeom)
        ]
        self.ng = max(len(g) for g in self.fgeom)
        self._f6 = np.zeros(6)
        self._vel6 = np.zeros(6)

    def _extent(self, g):
        p = self.m.geom_pos[g]
        s = self.m.geom_size[g]
        if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            return p[0] - s[0], p[0] + s[0]
        return p[0], p[0]

    def support_metrics(self, d: mujoco.MjData):
        m, fb, fgeom, ctr = self.m, self.fb, self.fgeom, self.ctr
        R = heading_rot(yaw_of(d.xquat[1]))
        com = d.subtree_com[1][:2]
        sole_w = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ ctr[k] for k, b in enumerate(fb)]
        toe_w = [
            d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([self.toe_x[k], ctr[k][1], ctr[k][2]])
            for k, b in enumerate(fb)
        ]
        F = np.zeros(2)
        copw = np.zeros((2, 3))
        gF = np.zeros((2, self.ng))  # per geom normal force
        gpen = np.zeros((2, self.ng))  # per geom max penetration (m, + = penetrating)
        quad = np.zeros((2, 4))  # per foot force: front, rear, inner, outer
        for i in range(d.ncon):
            c = d.contact[i]
            other = c.geom2 if c.geom1 == self.floor else (c.geom1 if c.geom2 == self.floor else -1)
            for k in range(2):
                if other not in fgeom[k]:
                    continue
                mujoco.mj_contactForce(m, d, i, self._f6)
                fn = self._f6[0]
                gi = fgeom[k].index(other)
                F[k] += fn
                copw[k] += fn * c.pos
                gF[k, gi] += fn
                gpen[k, gi] = max(gpen[k, gi], -c.dist)
                loc = d.xmat[fb[k]].reshape(3, 3).T @ (c.pos - d.xpos[fb[k]])
                quad[k, 0 if loc[0] > ctr[k][0] else 1] += fn
                inner = (loc[1] < ctr[k][1]) if k == 0 else (loc[1] > ctr[k][1])
                quad[k, 2 if inner else 3] += fn
        cop_loc = np.full(2, np.nan)
        cop_lat = np.full(2, np.nan)
        for k in range(2):
            if F[k] > 1:
                p = copw[k] / F[k]
                copw[k] = p
                loc = d.xmat[fb[k]].reshape(3, 3).T @ (p - d.xpos[fb[k]])
                cop_loc[k] = self.toe_x[k] - loc[0]
                cop_lat[k] = loc[1] - ctr[k][1]
        mid = 0.5 * (sole_w[0][:2] + sole_w[1][:2])
        com_h = R @ (com - mid)
        toe_front = max((R @ (t[:2] - com))[0] for t in toe_w)
        if F.sum() > 1:
            cop_h = R @ ((F[0] * copw[0][:2] + F[1] * copw[1][:2]) / F.sum() - mid)
        else:
            cop_h = np.full(2, np.nan)
        # per-geom tangential speed and sole pitch/roll
        gv = np.zeros((2, self.ng))
        for k in range(2):
            for gi, g in enumerate(fgeom[k]):
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, self._vel6, 0)
                gv[k, gi] = np.linalg.norm(self._vel6[3:5])
        sole = []
        for b in fb:
            Rb = d.xmat[b].reshape(3, 3)
            sole.append(
                (
                    np.arctan2(-Rb[2, 0], np.hypot(Rb[2, 1], Rb[2, 2])),
                    np.arctan2(Rb[2, 1], Rb[2, 2]),
                )
            )
        return com_h, toe_front, F, cop_loc, cop_lat, cop_h, gF, gpen, gv, quad, np.array(sole)


def main() -> None:
    args = parse_args()
    n_orig = resolve_n_orig(args.clip, args.n_orig)
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=build_overrides(args))

    import train_rsl_rl as T
    from rsl_rl.runners import OnPolicyRunner

    from unilab.training import create_env, ensure_registries

    ensure_registries()
    rl_cfg = T._algo_config_dict(cfg)
    env = create_env(cfg, num_envs=1, env_cfg_override=T.build_ppo_play_env_cfg_override(cfg))
    wrapped = T._resolve_ppo_wrapper_cls(rl_cfg)(env, device="cpu")
    train_cfg = T.normalize_ppo_train_cfg(rl_cfg)
    T.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    train_cfg.setdefault("runner", {})["logger"] = "none"
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
    load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False}
    runner.load(args.ckpt, load_cfg=load_cfg, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    ecfg = env._cfg
    be = env._backend
    m = be._model
    d = mujoco.MjData(m)
    dr_on = [k for k, v in vars(ecfg.domain_rand).items() if isinstance(v, bool) and v]
    pr, vr = ecfg.pose_randomization, ecfg.velocity_randomization
    print(
        "cfg check: sampling", ecfg.sampling_mode, "noise", ecfg.noise_config.level,
        "latency", ecfg.control_config.simulate_action_latency, "DR on:", dr_on,
        "ctrl_dt", ecfg.ctrl_dt, "sim_dt", ecfg.sim_dt, "model timestep", m.opt.timestep,
        "scene", ecfg.scene.model_file, "seed", args.seed,
        "pose_rand", {k: list(getattr(pr, k)) for k in AXES},
        "vel_rand", {k: list(getattr(vr, k)) for k in AXES},
    )  # fmt: skip
    scale = np.asarray(ecfg.control_config.action_scale, dtype=np.float64)
    default = np.asarray(env.default_angles, dtype=np.float64)

    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    dof = np.array([m.jnt_dofadr[j] for j in range(1, m.njnt)])
    qadr = np.array([m.jnt_qposadr[j] for j in range(1, m.njnt)])
    lim = np.array(
        [m.jnt_actfrcrange[j, 1] if m.jnt_actfrclimited[j] else np.inf for j in range(1, m.njnt)]
    )
    assert [m.actuator_trnid[a, 0] - 1 for a in range(m.nu)] == list(range(m.nu)), (
        "actuator order != joint order"
    )
    foot = FootModel(m)
    print(
        "foot geoms", [[mujoco.mj_id2name(m, 5, g) for g in gs] for gs in foot.fgeom],
        "toe/heel/ctr", foot.toe_x, foot.heel_x, foot.ctr,
        "solref", m.geom_solref[foot.fgeom[0][0]], "limits", {jn[i]: lim[i] for i in range(12)},
    )  # fmt: skip

    Z = np.load(args.clip)
    jp_ref, bq_ref, bp_ref = Z["joint_pos"], Z["body_quat_w"], Z["body_pos_w"]
    N = len(jp_ref)
    d.qpos[:] = 0
    d.qpos[:3] = bp_ref[n_orig - 1, 1]
    d.qpos[3:7] = bq_ref[n_orig - 1, 1]
    d.qpos[7:] = jp_ref[n_orig - 1]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    ref_com_h, ref_toe = foot.support_metrics(d)[:2]
    ref_rp = rp_of(bq_ref[n_orig - 1, 1])
    print(
        f"REF last frame (npz root): CoM ahead of mid-sole {ref_com_h[0] * 100:.1f}cm "
        f"lat {ref_com_h[1] * 100:.1f}cm toe margin {ref_toe * 100:.1f}cm "
        f"pelvis roll/pitch {np.degrees(ref_rp[0]):.1f}/{np.degrees(ref_rp[1]):.1f}deg"
    )

    obs, _ = wrapped.reset()
    assert int(env.motion_sampler.current_frames[0]) == 0
    rec = {k: [] for k in KEYS}
    with torch.inference_mode():
        for t in range(N - 2):
            frame = int(env.motion_sampler.current_frames[0])
            a = policy(obs)
            obs, _, done, _ = wrapped.step(a)
            a = a.numpy()[0].astype(np.float64)
            state = np.asarray(be._physics_state[0], dtype=np.float64)
            mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            d.ctrl[:] = default + a * scale
            mujoco.mj_forward(m, d)
            q = d.qpos[qadr].copy()
            com_h, toe_m, F, cop_loc, cop_lat, cop_h, gF, gpen, gv, quad, sole = (
                foot.support_metrics(d)
            )
            fr = min(frame + 1, N - 1)
            herr = abs(bp_ref[fr, 1, 2] - d.xpos[1][2])
            dq = abs(np.dot(bq_ref[fr, 1], d.xquat[1]))
            ori_sq = (2 * np.arccos(min(dq, 1))) ** 2
            vals = dict(
                frame=fr, tau=d.qfrc_actuator[dof].copy(), pderr=d.ctrl - q, act=a, q=q,
                qpos=d.qpos.copy(), com_h=com_h, toe_m=toe_m, F=F, cop_loc=cop_loc,
                cop_lat=cop_lat, cop_h=cop_h, gF=gF, gpen=gpen, gv=gv, quad=quad, sole=sole,
                rp=rp_of(d.xquat[1]), rp_ref=rp_of(bq_ref[fr, 1]), pel_xy=d.xpos[1][:2].copy(),
                pel_ref_xy=bp_ref[fr, 1, :2], pel_z=d.xpos[1][2],
                foot_xy=np.array([d.xpos[b][:2] for b in foot.fb]), done=bool(done[0]),
                trip=bool(herr > 0.45 or ori_sq > 0.8),
            )  # fmt: skip
            for k in KEYS:
                rec[k].append(vals[k])
            if done[0]:
                print("env reported done at step", t)
                break
    print("motion frames loaded", env.motion_loader.num_frames, "npz", N)
    R = {k: np.asarray(v) for k, v in rec.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, **R, jn=np.array(jn), lim=lim, n_orig=n_orig, ref_com_h=ref_com_h,
        ref_toe=ref_toe, ref_rp=np.array(ref_rp), toe_x=np.array(foot.toe_x),
        heel_x=np.array(foot.heel_x), seed=args.seed, pose_noise=args.pose_noise,
        vel_noise=args.vel_noise,
    )  # fmt: skip
    print("saved", args.out, "steps", len(R["frame"]))


if __name__ == "__main__":
    main()
