"""Binned trend of key metrics in an rsl_rl training log (stdout of train_rsl_rl.py).

usage:
    uv run --no-sync python utility/h2_hold/trend.py <train.log> [bin]

``bin`` = iterations averaged per row (default 10). Missing keys print as nan.
"""

from __future__ import annotations

import collections
import re
import sys

import numpy as np

K = [
    ("Mean episode length", "eplen"),
    ("Mean reward", "rew"),
    ("reward/joint_target_limit", "tgt_lim"),
    ("reward/joint_soft_limit", "soft_lim"),
    ("reward/motion_body_pos", "body_pos"),
    ("reward/motion_body_ori", "body_ori"),
    ("reward/motion_global_root_pos", "root_pos"),
    ("reward/upper_body_joint_pos", "upper"),
    ("reward/feet_slide", "slide"),
    ("reward/anti_shake_ang_vel", "shake"),
    ("termination/any", "term"),
    ("diagnostic/estimated_torque_saturation_ratio", "sat"),
    ("Mean value loss", "vloss"),
    ("Mean action std", "std"),
]


def parse(path: str) -> dict[int, dict[str, float]]:
    d: dict[int, dict[str, float]] = collections.defaultdict(dict)
    it = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.search(r"Learning iteration (\d+)/", line)
            if m:
                it = int(m.group(1))
                continue
            if it is None or ":" not in line:
                continue
            k, _, v = line.partition(":")
            try:
                d[it][k.strip()] = float(v.strip().rstrip("s"))
            except ValueError:
                pass
    return d


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__)
    d = parse(sys.argv[1])
    its = sorted(d)
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print("iter  " + " ".join(f"{n:>8}" for _, n in K))
    for i in range(0, len(its), b):
        s = its[i : i + b]
        vals = [np.mean([d[t].get(k, np.nan) for t in s]) for k, _ in K]
        print(f"{s[0]:5d} " + " ".join(f"{v:8.4f}" for v in vals))


if __name__ == "__main__":
    main()
