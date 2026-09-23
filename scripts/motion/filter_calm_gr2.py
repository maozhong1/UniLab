#!/usr/bin/env python3
"""Build a conservative GR2 motion subset for early curriculum training.

Unlike the H2 filter, this tool uses several independent limits so fast turns,
joint-speed spikes, jumps, and high kicks cannot pass solely because one aggregate
angular-velocity statistic is low. Inputs must be GR2-retargeted NPZ files produced
by ``g1_csv_to_gr2_npz.py`` and contain ``body_names`` metadata.

Examples:
  uv run --no-sync python scripts/motion/filter_calm_gr2.py \
      --src /path/to/npz_gr2 --dry-run
  uv run --no-sync python scripts/motion/filter_calm_gr2.py \
      --src /path/to/npz_gr2 --dst /path/to/npz_gr2_calm
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Limits:
    body_ang_p99: float
    joint_vel_p99: float
    base_lin_p99: float
    base_ang_p99: float
    base_z_range: float
    foot_height_max: float


@dataclass(frozen=True)
class Metrics:
    body_ang_p99: float
    joint_vel_p99: float
    base_lin_p99: float
    base_ang_p99: float
    base_z_range: float
    foot_height_max: float


def _percentile_magnitude(values: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(np.linalg.norm(values, axis=-1), percentile))


def _body_index(body_names: list[str], name: str) -> int:
    try:
        return body_names.index(name)
    except ValueError as error:
        raise ValueError(f"required body '{name}' is missing from body_names") from error


def clip_metrics(path: Path) -> Metrics:
    with np.load(path) as data:
        required = {
            "joint_vel",
            "body_pos_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "body_names",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing NPZ keys: {', '.join(missing)}")

        joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
        body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float64)
        body_names = [str(name) for name in data["body_names"].tolist()]

    arrays = (joint_vel, body_pos, body_lin_vel, body_ang_vel)
    if any(array.size == 0 for array in arrays):
        raise ValueError("motion arrays must not be empty")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("motion contains NaN or infinite values")

    base_index = _body_index(body_names, "base")
    left_foot_index = _body_index(body_names, "left_foot_roll_link")
    right_foot_index = _body_index(body_names, "right_foot_roll_link")
    base_z = body_pos[:, base_index, 2]
    foot_z = body_pos[:, (left_foot_index, right_foot_index), 2]

    return Metrics(
        body_ang_p99=_percentile_magnitude(body_ang_vel),
        joint_vel_p99=float(np.percentile(np.abs(joint_vel), 99.0)),
        base_lin_p99=_percentile_magnitude(body_lin_vel[:, base_index]),
        base_ang_p99=_percentile_magnitude(body_ang_vel[:, base_index]),
        base_z_range=float(np.ptp(base_z)),
        foot_height_max=float(np.max(foot_z)),
    )


def rejection_reasons(metrics: Metrics, limits: Limits) -> list[str]:
    reasons = []
    for field in Metrics.__dataclass_fields__:
        if getattr(metrics, field) > getattr(limits, field):
            reasons.append(field)
    return reasons


def _distribution(rows: list[tuple[Path, Metrics, list[str]]]) -> None:
    print("\nmetric distribution (per clip):")
    for field in Metrics.__dataclass_fields__:
        values = np.asarray([getattr(metrics, field) for _path, metrics, _reasons in rows])
        quantiles = np.percentile(values, (0, 25, 50, 75, 90, 95, 100))
        summary = ", ".join(
            f"{name}={value:.3f}"
            for name, value in zip(("min", "p25", "p50", "p75", "p90", "p95", "max"), quantiles)
        )
        print(f"  {field:18s} {summary}")


def _write_report(path: Path, rows: list[tuple[Path, Metrics, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Metrics.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(("file", "keep", "reasons", *fields))
        for source, metrics, reasons in rows:
            writer.writerow(
                (
                    source.name,
                    not reasons,
                    ";".join(reasons),
                    *(f"{getattr(metrics, field):.8g}" for field in fields),
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="directory of GR2 NPZ clips")
    parser.add_argument("--dst", help="calm subset directory (default: <src>_calm)")
    parser.add_argument("--body-ang-p99", type=float, default=8.0, metavar="RAD_S")
    parser.add_argument("--joint-vel-p99", type=float, default=7.0, metavar="RAD_S")
    parser.add_argument("--base-lin-p99", type=float, default=1.8, metavar="M_S")
    parser.add_argument("--base-ang-p99", type=float, default=2.5, metavar="RAD_S")
    parser.add_argument("--base-z-range", type=float, default=0.25, metavar="M")
    parser.add_argument("--foot-height-max", type=float, default=0.45, metavar="M")
    parser.add_argument("--copy", action="store_true", help="copy instead of symlink")
    parser.add_argument("--dry-run", action="store_true", help="scan without writing")
    parser.add_argument("--show-dropped", type=int, default=20)
    parser.add_argument("--report", help="CSV report path (default: <dst>/filter_report.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.src).expanduser().resolve()
    files = sorted(source_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no *.npz files under {source_dir}")

    limits = Limits(
        body_ang_p99=args.body_ang_p99,
        joint_vel_p99=args.joint_vel_p99,
        base_lin_p99=args.base_lin_p99,
        base_ang_p99=args.base_ang_p99,
        base_z_range=args.base_z_range,
        foot_height_max=args.foot_height_max,
    )
    rows = []
    invalid = []
    for path in files:
        try:
            metrics = clip_metrics(path)
            rows.append((path, metrics, rejection_reasons(metrics, limits)))
        except Exception as error:  # noqa: BLE001 - report each malformed clip and continue
            invalid.append((path, str(error)))

    if not rows:
        raise SystemExit("no valid GR2 NPZ clips found")

    kept = [row for row in rows if not row[2]]
    dropped = [row for row in rows if row[2]]
    print(f"scanned {len(rows)} valid clips from {source_dir}; invalid={len(invalid)}")
    print("limits:")
    for field in Limits.__dataclass_fields__:
        print(f"  {field:18s} <= {getattr(limits, field):g}")
    _distribution(rows)
    print(f"\nkeep {len(kept)}/{len(rows)} ({100.0 * len(kept) / len(rows):.1f}%); drop {len(dropped)}")

    reason_counts = {
        field: sum(field in reasons for _path, _metrics, reasons in dropped)
        for field in Metrics.__dataclass_fields__
    }
    print("drop reason counts:")
    for reason, count in reason_counts.items():
        print(f"  {reason:18s} {count}")
    if dropped:
        print(f"dropped clips (first {args.show_dropped}):")
        for path, _metrics, reasons in dropped[: args.show_dropped]:
            print(f"  {path.name}: {', '.join(reasons)}")
    for path, error in invalid:
        print(f"  INVALID {path.name}: {error}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return
    if not kept:
        raise SystemExit("limits keep zero clips; relax one or more thresholds")

    destination = (
        Path(args.dst).expanduser().resolve()
        if args.dst
        else source_dir.parent / f"{source_dir.name}_calm"
    )
    destination.mkdir(parents=True, exist_ok=True)
    for existing in destination.glob("*.npz"):
        if existing.is_symlink() or args.copy:
            existing.unlink()
        else:
            raise SystemExit(f"refusing to replace real file without --copy: {existing}")
    for source, _metrics, _reasons in kept:
        target = destination / source.name
        if args.copy:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source)

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else destination / "filter_report.csv"
    )
    _write_report(report_path, rows)
    mode = "copied" if args.copy else "symlinked"
    print(f"\n{mode} {len(kept)} clips -> {destination}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
