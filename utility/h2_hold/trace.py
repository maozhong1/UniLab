"""Per-step trace of one ``run_hold.py`` result around the clip end.

usage:
    uv run --no-sync python utility/h2_hold/trace.py <out.npz> [from_rel to_rel stride]

Rows are frames relative to the hold start (n_orig), default -25..200 every 5 frames.
Foot dx/dy are in cm, measured from the foot position at the hold start.
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__)
    Z = np.load(sys.argv[1])
    n0 = int(Z["n_orig"])
    jn = list(Z["jn"])
    fr = Z["frame"]
    a, b, s = (int(x) for x in (sys.argv[2:5] if len(sys.argv) > 4 else (-25, 200, 5)))
    J = {
        k: [jn.index(f"{side}_{k}_joint") for side in ("left", "right")]
        for k in ("ankle_pitch", "ankle_roll", "hip_roll", "knee", "hip_pitch")
    }
    AP, AR = J["ankle_pitch"], J["ankle_roll"]
    print(
        "rel | pitchErr | rollErr | comX comY | F_L F_R | copToe L R | apTau L R | arTau L R | "
        "arQ L R | arAct L R | apAct L R | apErr L R | footL dx,dy footR dx,dy (cm, from n0)"
    )
    i0 = min(int(np.searchsorted(fr, n0)), len(fr) - 1)
    f0 = Z["foot_xy"][i0]
    for i in range(len(fr)):
        rel = fr[i] - n0
        if rel < a or rel > b or rel % s:
            continue
        pe = np.degrees(Z["rp"][i] - Z["rp_ref"][i])
        t, q, ac, e = Z["tau"][i], Z["q"][i], Z["act"][i], Z["pderr"][i]
        fd = (Z["foot_xy"][i] - f0) * 100
        com, F, cop = Z["com_h"][i] * 100, Z["F"][i], Z["cop_loc"][i] * 100
        print(
            f"{rel:+4d} | {pe[1]:+5.1f} | {pe[0]:+5.1f} | {com[0]:+5.1f} {com[1]:+5.1f} | "
            f"{F[0]:4.0f} {F[1]:4.0f} | {cop[0]:5.1f} {cop[1]:5.1f} | "
            f"{t[AP][0]:+5.1f} {t[AP][1]:+5.1f} | {t[AR][0]:+5.1f} {t[AR][1]:+5.1f} | "
            f"{q[AR][0]:+.2f} {q[AR][1]:+.2f} | {ac[AR][0]:+5.1f} {ac[AR][1]:+5.1f} | "
            f"{ac[AP][0]:+5.2f} {ac[AP][1]:+5.2f} | {e[AP][0]:+.2f} {e[AP][1]:+.2f} | "
            f"{fd[0][0]:+5.1f},{fd[0][1]:+5.1f} {fd[1][0]:+5.1f},{fd[1][1]:+5.1f}"
        )


if __name__ == "__main__":
    main()
