"""Hold-test report for a directory of ``run_hold.py`` result npz files.

Importable: ``load(path)`` -> dict, ``hold_stats(Z, phase)`` -> metric dict, with phase in
``hold`` (frames after the original clip end), ``last1s`` (last second of the clip) or ``clip``.

usage:
    uv run --no-sync python utility/h2_hold/analyze.py <result_dir> [<variant_dir>]

``<variant_dir>`` holds ``<clip>__<tag>.npz`` files (contact-model A/B variants or perturbed
seeds from ``run_suite.sh``); each is printed next to ``<result_dir>/<clip>.npz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

D = np.degrees
CTRL_DT = 0.02  # s per recorded step (policy rate)
LIFT_N = 20.0  # N: a foot "lifts" when its normal force drops below this
KNEE_LOCK_Q = -0.08  # rad: knee below this sits at its hyper-extension stop (-0.087)
SAT_FRAC = 0.98  # |tau| >= SAT_FRAC * limit counts as saturated


def load(f: str | Path) -> dict:
    Z = dict(np.load(f))
    Z["name"] = Path(f).stem
    return Z


def hold_stats(Z: dict, ph: str = "hold") -> dict:
    n0 = int(Z["n_orig"])
    fr = Z["frame"]
    jn = list(Z["jn"])
    lim = Z["lim"]
    sel = {"hold": fr >= n0, "last1s": (fr >= n0 - 50) & (fr < n0), "clip": fr < n0}[ph]

    def J(s):
        return [jn.index(f"{x}_{s}_joint") for x in ("left", "right")]

    AP, AR, KN = J("ankle_pitch"), J("ankle_roll"), J("knee")
    tau, err, act, F = Z["tau"][sel], Z["pderr"][sel], Z["act"][sel], Z["F"][sel]

    def sat(j):
        return 100 * np.mean(np.abs(tau[:, j]) >= SAT_FRAC * lim[j])

    perr = D(Z["rp"][sel, 1] - Z["rp_ref"][sel, 1])
    com = Z["com_h"][sel] * 100
    cl = Z["cop_loc"][sel] * 100
    w = np.where(np.isnan(cl), 0, F)
    clw = np.nansum(np.nan_to_num(cl) * w, 1) / np.maximum(w.sum(1), 1e-6)
    fx = Z["foot_xy"][sel]
    disp = np.linalg.norm(fx - fx[0], axis=2).max(0) * 100
    lifts = sum(int(np.sum((F[1:, k] < LIFT_N) & (F[:-1, k] >= LIFT_N))) for k in range(2))
    r = dict(
        perr_mean=perr.mean(),
        perr_max=perr[np.argmax(np.abs(perr))],
        com_mean=com[:, 0].mean(),
        com_max=com[:, 0].max(),
        comy=com[:, 1].mean(),
        cop_toe=np.mean(clw),
        cop_toe_min=np.nanmin(cl) if np.isfinite(cl).any() else np.nan,
        apT_L=tau[:, AP[0]].mean(),
        apT_R=tau[:, AP[1]].mean(),
        apT_max=np.abs(tau[:, AP]).max(),
        apS=sat(AP),
        arS_L=sat(AR[0]),
        arS_R=sat(AR[1]),
        apE_L=err[:, AP[0]].mean(),
        apE_R=err[:, AP[1]].mean(),
        apE_max=np.abs(err[:, AP]).max(),
        apA_L=act[:, AP[0]].mean(),
        apA_R=act[:, AP[1]].mean(),
        arA_L=act[:, AR[0]].mean(),
        arA_R=act[:, AR[1]].mean(),
        knee_lock=100 * np.mean(Z["q"][sel][:, KN] < KNEE_LOCK_Q),
        knE=err[:, KN].mean(),
        lifts=lifts,
        disp=disp.max(),
        drift=np.linalg.norm(Z["pel_xy"][sel][-1] - Z["pel_xy"][sel][0]) * 100,
        fall=int((Z["pel_z"][sel] < 0.5).any()),
        trip=int(Z["trip"][sel].any()),
        FL=F[:, 0].mean(),
        FR=F[:, 1].mean(),
    )
    # contact detail
    gF, gv, quad, sole = Z["gF"][sel], Z["gv"][sel], Z["quad"][sel], Z["sole"][sel]
    Ft = np.maximum(F, 1)
    r["front_frac"] = 100 * np.mean(quad[:, :, 0].sum(1) / np.maximum(F.sum(1), 1))
    loaded = F > 100
    r["heel_lift"] = 100 * np.mean((quad[:, :, 1] < 0.05 * Ft)[loaded]) if loaded.any() else np.nan
    r["outer_frac_R"] = 100 * np.mean(quad[:, 1, 3] / Ft[:, 1])
    r["outer_frac_L"] = 100 * np.mean(quad[:, 0, 3] / Ft[:, 0])
    r["sole_pitch_std"] = D(sole[:, :, 0]).std(0).max()
    r["sole_pitch_rng"] = np.ptp(D(sole[:, :, 0]), 0).max()
    r["sole_roll_L"] = D(sole[:, 0, 1]).mean()
    r["sole_roll_R"] = D(sole[:, 1, 1]).mean()
    r["pen_max"] = Z["gpen"][sel].max() * 1000
    r["pen_mean"] = Z["gpen"][sel][gF > 20].mean() * 1000 if (gF > 20).any() else 0
    on = gF > 1
    T = sel.sum() * CTRL_DT
    r["chatter"] = np.sum(on[1:] != on[:-1]) / T / (on.shape[1] * on.shape[2])  # /s/geom
    ld = gF > 20
    r["slip_v"] = gv[ld].mean() if ld.any() else 0
    r["slip_pct"] = 100 * np.mean(gv[ld] > 0.05) if ld.any() else 0
    return r


def _hold_row(r: dict) -> str:
    return (
        f"{r['perr_mean']:+.1f}/{r['perr_max']:+.1f} | {r['com_mean']:+.1f}/{r['com_max']:+.1f} | "
        f"{r['cop_toe']:.1f}/{r['cop_toe_min']:.1f} | "
        f"{r['apT_L']:+.0f}/{r['apT_R']:+.0f}, {r['apT_max']:.0f} | {r['apS']:.0f} | "
        f"{r['arS_L']:.0f}/{r['arS_R']:.0f} | "
        f"{r['apE_L']:+.2f}/{r['apE_R']:+.2f} ({r['apE_max']:.2f}) | "
        f"{r['apA_L']:+.1f}/{r['apA_R']:+.1f} | {r['arA_L']:+.1f}/{r['arA_R']:+.1f} | "
        f"{r['knee_lock']:.0f} | {r['lifts']} | {r['disp']:.0f} | {r['drift']:.0f} | "
        f"{r['fall']} | {r['trip']} | {r['FL']:.0f}/{r['FR']:.0f}"
    )


def report(result_dir: Path, variant_dir: Path | None = None) -> None:
    base = {Z["name"]: Z for Z in map(load, sorted(result_dir.glob("*.npz")))}
    if not base:
        raise SystemExit(f"no *.npz in {result_dir}")
    print(f"## Hold phase ({result_dir})")
    print(
        "clip | pitchErr mean/max deg | CoM-ahead mean/max cm | CoP-toe wmean/min cm | "
        "apTau L/R mean, max | apSat% | arSat% L/R | apErr L/R (max) rad | apAct L/R | "
        "arAct L/R | kneeLock% | lifts | footDisp cm | drift cm | fall | trip | F L/R N"
    )
    for k, Z in base.items():
        print(f"{k[:26]} | {_hold_row(hold_stats(Z))}")

    print("\n## Last 1 s of clip (before hold)")
    for k, Z in base.items():
        r = hold_stats(Z, "last1s")
        print(
            f"{k[:26]} | pitchErr {r['perr_mean']:+.1f}/{r['perr_max']:+.1f} | "
            f"CoM {r['com_mean']:+.1f}/{r['com_max']:+.1f} | CoP-toe {r['cop_toe']:.1f} | "
            f"apTau {r['apT_L']:+.0f}/{r['apT_R']:+.0f} | "
            f"apErr {r['apE_L']:+.2f}/{r['apE_R']:+.2f} | arSat R {r['arS_R']:.0f} | "
            f"kneeLock {r['knee_lock']:.0f}"
        )

    print(
        "\n## Reference hold pose (npz last frame, npz root pose): "
        "CoM ahead of mid-sole / CoM-to-front-toe margin (cm)"
    )
    for k, Z in base.items():
        print(f"{k[:26]} {Z['ref_com_h'][0] * 100:+.1f} / {Z['ref_toe'] * 100:.1f}")

    print("\n## Contact detail (hold)")
    print(
        "clip | front-half force % | heel-unloaded % (loaded feet) | outer-edge force % L/R | "
        "sole roll L/R deg | sole pitch std/range deg | pen max/mean mm | chatter /s/geom | "
        "slip v m/s, %>5cm/s"
    )
    for k, Z in base.items():
        r = hold_stats(Z)
        print(
            f"{k[:26]} | {r['front_frac']:.0f} | {r['heel_lift']:.0f} | "
            f"{r['outer_frac_L']:.0f}/{r['outer_frac_R']:.0f} | "
            f"{r['sole_roll_L']:+.1f}/{r['sole_roll_R']:+.1f} | "
            f"{r['sole_pitch_std']:.2f}/{r['sole_pitch_rng']:.1f} | "
            f"{r['pen_max']:.1f}/{r['pen_mean']:.1f} | {r['chatter']:.2f} | "
            f"{r['slip_v']:.3f}, {r['slip_pct']:.0f}"
        )

    var = sorted(variant_dir.glob("*.npz")) if variant_dir else []
    if not var:
        return
    print(f"\n## Variants (hold) ({variant_dir})")
    print(
        "clip variant | pitchErr mean/max | CoM mean/max | CoP-toe | apTau L/R | arSat R | "
        "kneeLock | lifts | footDisp | drift | fall | front% | heelUnl% | sole roll R | "
        "chatter | slip%"
    )
    shown = set()
    for f in var:
        Z = load(f)
        if "__" not in Z["name"]:
            continue
        b, v = Z["name"].rsplit("__", 1)
        rows = [(v, Z)]
        if b not in shown and b in base:
            rows.insert(0, ("base", base[b]))
            shown.add(b)
        for tag, ZZ in rows:
            r = hold_stats(ZZ)
            print(
                f"{b[:34]:34s} {tag:6s} | {r['perr_mean']:+.1f}/{r['perr_max']:+.1f} | "
                f"{r['com_mean']:+.1f}/{r['com_max']:+.1f} | {r['cop_toe']:.1f} | "
                f"{r['apT_L']:+.0f}/{r['apT_R']:+.0f} | {r['arS_R']:.0f} | "
                f"{r['knee_lock']:.0f} | {r['lifts']} | {r['disp']:.0f} | {r['drift']:.0f} | "
                f"{r['fall']} | {r['front_frac']:.0f} | {r['heel_lift']:.0f} | "
                f"{r['sole_roll_R']:+.1f} | {r['chatter']:.2f} | {r['slip_pct']:.0f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("variant_dir", type=Path, nargs="?", default=None)
    args = ap.parse_args()
    report(args.result_dir, args.variant_dir)


if __name__ == "__main__":
    main()
