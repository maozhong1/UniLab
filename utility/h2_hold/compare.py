"""Compare hold-phase stats of two ``run_hold.py`` result dirs, clip by clip.

usage:
    uv run --no-sync python utility/h2_hold/compare.py <dirA> <dirB> [labelA labelB]

Each cell prints ``A>B`` for clips present in both dirs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import hold_stats, load

KEYS = [
    ("perr_mean", "pitch err"),
    ("com_max", "CoM max"),
    ("cop_toe", "CoP toe"),
    ("apE_R", "apErr R"),
    ("apS", "ap sat%"),
    ("arS_R", "arR sat%"),
    ("arA_R", "arR act"),
    ("knee_lock", "knee lock%"),
    ("lifts", "lifts"),
    ("disp", "foot disp"),
]


def compare(dir_a: Path, dir_b: Path, la: str = "A", lb: str = "B") -> None:
    print("clip".ljust(16), " ".join(f"{n:>15}" for _, n in KEYS))
    n = 0
    for fb in sorted(dir_b.glob("*.npz")):
        fa = dir_a / fb.name
        if not fa.exists():
            continue
        a, b = hold_stats(load(fa)), hold_stats(load(fb))
        print(fb.name[:16].ljust(16), " ".join(f"{a[k]:7.1f}>{b[k]:<7.1f}" for k, _ in KEYS))
        n += 1
    if not n:
        print(f"(no common *.npz between {dir_a} and {dir_b})")
    print(f"(each cell: {la} > {lb})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dir_a", type=Path)
    ap.add_argument("dir_b", type=Path)
    ap.add_argument("label_a", nargs="?", default="A")
    ap.add_argument("label_b", nargs="?", default="B")
    args = ap.parse_args()
    compare(args.dir_a, args.dir_b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
