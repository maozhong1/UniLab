"""Select upright BONES-SEED operation clips for GR2 retargeting.

Selection combines metadata exclusions with measured G1 posture. It rejects
locomotion, low/floor work, sitting, crouching, passive holding, and looking,
then limits root-height variation, waist pitch, and knee flexion. Remaining
clips are ranked by active shoulder/elbow/wrist motion and operation semantics.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from unilab.tools.bones_seed_csv import load_header, parse_joint_names

_ALLOWED_CATEGORIES = {
    "Object Manipulation",
    "Object Interaction",
    "Household",
    "Consuming",
    "Complex Actions",
}
_EXCLUDED_SEMANTICS = re.compile(
    r"\b(floor|ground|crouch(?:ing)?|croach(?:ing)?|kneel(?:ing)?|sit(?:ting)?|"
    r"walk(?:ing)?|jog(?:ging)?|run(?:ning)?|throw(?:ing)?|catch(?:ing)?|"
    r"climb(?:ing)?|jump(?:ing)?|low|idle|hold(?:ing|s)?|look(?:ing|s)?)\b|"
    r"pick[ _]?up",
    re.IGNORECASE,
)
_OPERATION_SEMANTICS = re.compile(
    r"\b(table|desk|shelf|cupboard|fridge|microwave|blender|dish|food|bread|"
    r"vegetable|cook|cut|chop|peel|grate|wash|clean|wipe|nail|hammer|saw|"
    r"tool|button|lever|valve|door|phone|type|write|sew|iron|assemble|repair|"
    r"inspect|operate|open|close|take out|put away|put down|give|receive)\b",
    re.IGNORECASE,
)
_TRIAL_SUFFIX = re.compile(r"_\d{3}__.*$")
_PROXIMAL_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
_WRIST_JOINTS = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_POSTURE_JOINTS = (
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
)


@dataclass(frozen=True)
class Candidate:
    row: dict[str, str]
    source: Path
    family: str
    root_z_range_cm: float
    waist_pitch_abs_p95_deg: float
    knee_abs_p95_deg: float
    proximal_excursion_deg: float
    wrist_excursion_deg: float
    operation_priority: bool
    score: float


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _metadata_text(row: dict[str, str]) -> str:
    fields = (
        "move_name",
        "content_short_description",
        "content_natural_desc_1",
        "content_natural_desc_2",
        "content_props",
    )
    return " ".join(row[field] for field in fields).replace("_", " ")


def _action_family(move_name: str) -> str:
    return _TRIAL_SUFFIX.sub("", move_name)


def _clip_metrics(csv_file: Path) -> tuple[float, float, float, float, float]:
    joint_names = parse_joint_names(load_header(csv_file), csv_file)
    columns = {name: index + 7 for index, name in enumerate(joint_names)}
    names = _PROXIMAL_JOINTS + _WRIST_JOINTS + _POSTURE_JOINTS
    usecols = [3, *(columns[name] for name in names)]
    values = np.loadtxt(csv_file, delimiter=",", skiprows=1, usecols=usecols, ndmin=2)

    root_z = values[:, 0]
    joints = values[:, 1:]
    arm_count = len(_PROXIMAL_JOINTS) + len(_WRIST_JOINTS)
    arm_range = np.percentile(joints[:, :arm_count], 95, axis=0) - np.percentile(
        joints[:, :arm_count], 5, axis=0
    )
    proximal = float(np.mean(arm_range[: len(_PROXIMAL_JOINTS)]))
    wrist = float(np.mean(arm_range[len(_PROXIMAL_JOINTS) :]))
    posture = joints[:, arm_count:]
    return (
        float(np.percentile(root_z, 95) - np.percentile(root_z, 5)),
        float(np.percentile(np.abs(posture[:, 0]), 95)),
        float(
            max(
                np.percentile(np.abs(posture[:, 1]), 95),
                np.percentile(np.abs(posture[:, 2]), 95),
            )
        ),
        proximal,
        wrist,
    )


def _load_candidates(args: argparse.Namespace) -> list[Candidate]:
    with args.metadata.open(newline="", encoding="utf-8") as metadata_file:
        rows = []
        for row in csv.DictReader(metadata_file):
            text = _metadata_text(row)
            if row["category"] not in _ALLOWED_CATEGORIES:
                continue
            if _is_true(row["is_mirror"]) or row["content_body_position"] != "standing":
                continue
            if _EXCLUDED_SEMANTICS.search(text):
                continue
            frames = int(row["move_duration_frames"])
            if args.min_frames <= frames <= args.max_frames:
                rows.append(row)

    candidates = []
    for row in tqdm(rows, desc="Scoring upright operation clips"):
        source = args.dataset_root / row["move_g1_path"]
        if not source.is_file():
            raise FileNotFoundError(f"G1 CSV not found: {source}")
        root_z, waist_pitch, knee, proximal, wrist = _clip_metrics(source)
        if root_z > args.max_root_z_range_cm:
            continue
        if waist_pitch > args.max_waist_pitch_deg or knee > args.max_knee_deg:
            continue
        if proximal < args.min_proximal_excursion_deg:
            continue
        priority = bool(_OPERATION_SEMANTICS.search(_metadata_text(row)))
        score = proximal + 0.25 * wrist + (20.0 if priority else 0.0)
        candidates.append(
            Candidate(
                row=row,
                source=source,
                family=_action_family(row["move_name"]),
                root_z_range_cm=root_z,
                waist_pitch_abs_p95_deg=waist_pitch,
                knee_abs_p95_deg=knee,
                proximal_excursion_deg=proximal,
                wrist_excursion_deg=wrist,
                operation_priority=priority,
                score=score,
            )
        )
    return candidates


def _select(
    candidates: list[Candidate], count: int, per_family: int, per_actor: int
) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda item: (-item.score, item.source.as_posix()))
    family_counts: dict[str, int] = {}
    actor_counts: dict[str, int] = {}
    selected = []
    for candidate in ranked:
        actor = candidate.row["actor_uid"]
        if family_counts.get(candidate.family, 0) >= per_family:
            continue
        if actor_counts.get(actor, 0) >= per_actor:
            continue
        selected.append(candidate)
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
        actor_counts[actor] = actor_counts.get(actor, 0) + 1
        if len(selected) == count:
            break
    return selected


def _write_manifest(selected: list[Candidate], output: Path, dataset_root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "rank", "source_csv", "move_name", "action_family", "category", "actor_uid",
        "frames", "duration_seconds", "root_z_range_cm", "waist_pitch_abs_p95_deg",
        "knee_abs_p95_deg", "proximal_excursion_deg", "wrist_excursion_deg",
        "operation_priority", "selection_score", "description", "props",
    )
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for rank, candidate in enumerate(selected, start=1):
            row = candidate.row
            frames = int(row["move_duration_frames"])
            writer.writerow(
                {
                    "rank": rank,
                    "source_csv": candidate.source.relative_to(dataset_root).as_posix(),
                    "move_name": row["move_name"],
                    "action_family": candidate.family,
                    "category": row["category"],
                    "actor_uid": row["actor_uid"],
                    "frames": frames,
                    "duration_seconds": f"{frames / 120.0:.3f}",
                    "root_z_range_cm": f"{candidate.root_z_range_cm:.3f}",
                    "waist_pitch_abs_p95_deg": f"{candidate.waist_pitch_abs_p95_deg:.3f}",
                    "knee_abs_p95_deg": f"{candidate.knee_abs_p95_deg:.3f}",
                    "proximal_excursion_deg": f"{candidate.proximal_excursion_deg:.3f}",
                    "wrist_excursion_deg": f"{candidate.wrist_excursion_deg:.3f}",
                    "operation_priority": candidate.operation_priority,
                    "selection_score": f"{candidate.score:.3f}",
                    "description": row["content_short_description"],
                    "props": row["content_props"],
                }
            )


def _write_link_tree(selected: list[Candidate], link_dir: Path, dataset_root: Path) -> None:
    for candidate in selected:
        relative = candidate.source.relative_to(dataset_root / "g1" / "csv")
        destination = link_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() and destination.resolve() == candidate.source:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to replace existing path: {destination}")
        destination.symlink_to(candidate.source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-dir", type=Path)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--per-family", type=int, default=24)
    parser.add_argument("--per-actor", type=int, default=120)
    parser.add_argument("--min-frames", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--max-root-z-range-cm", type=float, default=8.0)
    parser.add_argument("--max-waist-pitch-deg", type=float, default=20.0)
    parser.add_argument("--max-knee-deg", type=float, default=35.0)
    parser.add_argument("--min-proximal-excursion-deg", type=float, default=8.0)
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.metadata = args.metadata.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.link_dir is not None:
        args.link_dir = args.link_dir.expanduser().resolve()

    candidates = _load_candidates(args)
    selected = _select(candidates, args.count, args.per_family, args.per_actor)
    _write_manifest(selected, args.output, args.dataset_root)
    if args.link_dir is not None:
        _write_link_tree(selected, args.link_dir, args.dataset_root)
    print(f"Selected {len(selected)}/{len(candidates)} eligible clips into {args.output}")


if __name__ == "__main__":
    main()