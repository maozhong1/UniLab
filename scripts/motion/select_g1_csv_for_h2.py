"""Select a G1 BONES-SEED CSV subset for G1 -> H2 retargeting (``g1_csv_to_h2_npz2.py``).

The full BONES-SEED G1 set (~142k clips, every motion repeated over many performers and
mirrored as ``*_M.csv``) contains sitting/crawling/prop-dependent motions and very dynamic or
acrobatic ones that H2 cannot track. This script reproduces the selection used for the H2
tracking data set, from per-clip kinematic statistics plus name rules:

1. **Stats** (``--stats_cache`` makes this a one-off): per clip, pelvis height / torso tilt /
   root & joint speeds, from the raw 120 fps CSV (root in cm, angles in degrees).
2. **Keyword exclusion** (all groups): any motion whose name contains one of
   ``EXCLUDE_KEYWORDS`` (terrain / furniture / props, acrobatics, paper extras) is dropped first.
3. **General filter** (``other`` motions): drops sit/chair/box/terrain, crawl/ground/kneel,
   acrobatic/jump, fall/injury/prop-dependent names, then kinematic gates (pelvis low or
   airborne, torso tilt, root / vertical / joint speed, clip length).
4. **Per-group gates**: ``dance`` / ``jump`` / ``kneel`` motions are judged by their own, more
   permissive gates (dances move fast; jumps leave the floor; kneeling is low) while still
   rejecting flips, ground work, terrain and props.
5. **Emote removal**: pure gesture / daily-activity motions (itching, looking around, ...) are
   dropped; dance, jump, kneel, locomotion and the ``--keep_always`` motions stay.
6. **Per-motion cap** (``--max_per_motion``): a motion is one filename stem (e.g.
   ``dancing_routine_V002_001``). Originals are spread evenly over performers and each picked
   original keeps its ``_M`` mirror, so a capped motion has at most ``cap`` clips.

Outputs in ``--output_dir``: ``selected.txt`` (absolute CSV paths), ``manifest.csv`` (every
clip with group / category / decision / reason / stats) and ``summary.txt``. ``--link_dir``
additionally builds a flat directory of symlinks, which is what ``g1_csv_to_h2_npz2.py
--input`` consumes.

Usage:
    uv run scripts/motion/select_g1_csv_for_h2.py --input <g1_csv_dir> --output_dir <out> \
        --stats_cache <out>/stats.csv --link_dir <g1_csv_filtered> --workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

Record = dict[str, Any]

STAT_FIELDS: tuple[str, ...] = (
    "n_frames",
    "dur",
    "z_min",
    "z_p5",
    "z_med",
    "z_max",
    "z_range",
    "up_min",
    "up_p5",
    "speed_max",
    "vz_p99",
    "qd_p99",
    "qd_max",
)

# --------------------------------------------------------------------------------------
# Name rules. Matching is on the lower-cased stem (filename without ``__A###[_M].csv``).
# --------------------------------------------------------------------------------------


def _tok(words: list[str]) -> re.Pattern[str]:
    """Whole-token match: word delimited by ``_`` / start / end / a trailing digit."""
    return re.compile(r"(?:^|_)(?:" + "|".join(words) + r")(?:_|$|\d)")


# General rules, first hit wins (applied to motions that belong to no special group).
_GENERAL_NAME_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "name:crawl/ground/kneel",
        _tok(
            [
                "crawl[a-z]*",
                "fours",
                "lying",
                "fetal",
                "kneel[a-z]*",
                "sit_on_heels",
                "plank",
                "worm",
                "inchworm",
                "crab",
                "swimming",
                "breaststroke",
                "scuttle",
                "snake",
                "ground",
                "floor",
                "monkeys?",
                "spider",
                "frog",
                "tiger",
                "cat",
                "camel",
                "pushups?",
                "dips?",
                "stretch[a-z]*_floor",
            ]
        ),
    ),
    (
        "name:sit/chair/box/terrain",
        _tok(
            [
                "sit[a-z]*",
                "seat",
                "chair",
                "bench",
                "box",
                "crate",
                "table",
                "stairs?",
                "upstairs",
                "downstairs",
                "ladder[a-z]*",
                "climb[a-z]*",
                "climber",
                "mountain",
                "roof",
                "wall",
                "bus",
                "car",
                "taxi",
                "passenger",
                "horse",
                "riding",
                "ridding",
                "ride",
                "bicycle",
                "bar",
                "stadium",
                "stump",
                "stepdown",
                "obstacle",
                "fence",
                "hole",
                "edge",
                "rop",
                "rope",
            ]
        ),
    ),
    (
        "name:acrobatic/flip/jump",
        _tok(
            [
                "jump[a-z]*",
                "jumps",
                "jumping",
                "jumpdown",
                "hop",
                "flips?",
                "cartwheel[a-z]*",
                "handstand",
                "hspu",
                "vault",
                "burpee",
                "superman",
                "somersault",
                "breakdance",
                "land",
                "landing",
                "fly",
                "flying",
                "ninja",
                "sprint",
                "handspring",
                "spin",
                "pirouette",
                "dive",
                "diagonal",
            ]
        ),
    ),
    (
        "name:fall/injury/prop-dep",
        _tok(
            [
                "fall",
                "faint",
                "convulsions?",
                "death",
                "postmortem",
                "trip",
                "slip",
                "carcrash",
                "crutch[a-z]*",
                "wheels",
                "skipping",
                "tuberculosis",
                "explosion",
                "extreme",
                "rage",
                "chase",
                "strafing",
                "punch[a-z]*",
                "boxing",
                "combat",
                "dodge",
                "slam",
                "kick[a-z]*",
            ]
        ),
    ),
)

# General kinematic gates (a clip is dropped when ``bad(stats)`` is true), first hit wins.
_GENERAL_GATES: tuple[tuple[str, Any], ...] = (
    ("kin:dur<1.5s or >60s", lambda s: s["dur"] < 1.5 or s["dur"] > 60),
    ("kin:pelvis low (p5<0.55m)", lambda s: s["z_p5"] < 0.55),
    ("kin:pelvis min<0.40m", lambda s: s["z_min"] < 0.40),
    (
        "kin:pelvis high/airborne (zmax>1.0 or range>0.35)",
        lambda s: s["z_max"] > 1.0 or s["z_range"] > 0.35,
    ),
    ("kin:torso tilt (up_p5<0.85 or up_min<0.6)", lambda s: s["up_p5"] < 0.85 or s["up_min"] < 0.6),
    ("kin:root speed>3.0m/s", lambda s: s["speed_max"] > 3.0),
    ("kin:vertical speed p99>0.8m/s", lambda s: s["vz_p99"] > 0.8),
    ("kin:joint vel p99>450deg/s", lambda s: s["qd_p99"] > 450),
    ("kin:joint vel max>1500deg/s", lambda s: s["qd_max"] > 1500),
)

# Special groups. A motion matching several groups takes the last one: kneel < jump < dance.
_DANCE_RE = re.compile(
    r"dance|macarena|charleston|egypt|gangam|ymca|griddy|floss|moonwalk|twist|disco|salsa|mambo"
    r"|jive|krakowiak|kozak|(?:^|_)dab(?:_|$)|cosita|toosie|dutty|dames|kabanse|padeburee"
    r"|shuffle|jazz|chaines|quickstep|choreograph|hiphop|vouge|latino|funky|boogle|wutang"
)
_JUMP_RE = _tok(["jump[a-z]*", "jumps", "jumping", "hop", "frog"])
_JUMP_EXTRA_RE = re.compile(r"frog_jump|one_leg_jump")
_KNEEL_RE = _tok(["kneel[a-z]*"])

# Names rejected inside the dance / kneel groups (flips, ground work, terrain, falls).
_HARD_BAD_RE = _tok(
    [
        "flips?",
        "cartwheel[a-z]*",
        "handstand",
        "hspu",
        "vault",
        "somersault",
        "breakdance",
        "worm",
        "inchworm",
        "crawl[a-z]*",
        "fours",
        "lying",
        "fetal",
        "plank",
        "floor",
        "ground",
        "sit[a-z]*",
        "seat",
        "chair",
        "bench",
        "box",
        "crate",
        "stairs?",
        "ladder[a-z]*",
        "climb[a-z]*",
        "obstacle",
        "fence",
        "wall",
        "roof",
        "horse",
        "riding",
        "ride",
        "bicycle",
        "car",
        "bus",
        "jumpdown",
        "stepdown",
        "carcrash",
        "fall",
        "faint",
        "death",
        "postmortem",
        "convulsions?",
    ]
)
# Jumps are allowed, but not onto / over / off things.
_JUMP_BAD_RE = _tok(
    [
        "obstacle",
        "ladder[a-z]*",
        "box",
        "crate",
        "stairs?",
        "climb[a-z]*",
        "jumpdown",
        "fence",
        "wall",
        "flips?",
        "cartwheel[a-z]*",
        "handstand",
        "vault",
        "somersault",
    ]
)
_JUMP_BAD_EXTRA_RE = re.compile(r"jump_on|jump_off|jump_down|jump_over")
_FROG_RE = re.compile(r"frog")
_FROG_Z_RANGE = 0.80  # frog jumps start from a deep squat, so their pelvis-height range is large

_GROUP_GATES: dict[str, dict[str, Any]] = {
    "dance": dict(
        dur=(1.5, 60), z_p5=0.50, zmax=1.10, zr=0.50, up=0.75, sp=3.5, vz=1.5, qd99=800, qdmax=2500
    ),
    "jump": dict(
        dur=(1.5, 60), z_p5=0.15, zmax=1.15, zr=0.45, up=0.80, sp=3.5, vz=2.5, qd99=800, qdmax=3500
    ),
    "kneel": dict(
        dur=(1.5, 60), z_p5=0.20, zmax=1.00, zr=0.55, up=0.70, sp=1.5, vz=0.8, qd99=450, qdmax=1500
    ),
}

# Category labels (first match wins). Only ``gesture/emote`` (the catch-all) is removed.
_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("dance_hiphop", "dance_hiphop"),
    ("dance_vouge", "dance_vouge"),
    ("dance_other", "dance"),
    ("walk", "walk"),
    ("jog/run", "jog|run"),
    ("idle/turn", "idle|turn|step_rotate"),
    ("door/lever/valve/crank", "door|lever|valve|crank|switch|knob|handle"),
    ("pick/put/carry", "pick|put|lift|take|carry|grab|pass"),
    ("injured", "injur|inj_"),
    ("gesture/emote", "."),
)
_EMOTE = "gesture/emote"
# Not emotes even though they fall in the catch-all category: low/crouched locomotion, and the
# long ``dancing_routine`` takes (named "dancing", so the dance regex does not see them).
_LOCOMOTION_RE = re.compile(r"^(?:crouch_ff|h_b_w_)")
_DANCING_ROUTINE_RE = re.compile(r"^dancing_routine")

# Substring keywords that exclude a motion in every group (terrain / furniture / props / bikes,
# acrobatics, and extras from the paper's data cleaning). Matched on the lower-cased stem.
EXCLUDE_KEYWORDS_SONIC: tuple[str, ...] = (
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table",
    "_sit_", "sit_", "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "box_hspu",
    "50cm_box_", "on_box", "fall_from", "on_1m", "form_box", "off_1m", "230m",
    "jump_over_obstacle_", "lift_crate_come_up_", "stair", "box_jump", "box_dips",
    "walking_on_edge", "push_obstacle",
)  # fmt: skip
EXCLUDE_KEYWORDS_ACROBATICS: tuple[str, ...] = (
    "acrobatics_", "cartwheel", "handstand", "handstand_ff_", "safety_roll", "kozak_dance",
    "jump_to_shoulder_roll", "monkey_jump",
)  # fmt: skip
EXCLUDE_KEYWORDS_PAPER: tuple[str, ...] = (
    "_box_", "box_", "_stair", "jump_off_50cm", "off_50cm", "fence", "look_over",
)  # fmt: skip
EXCLUDE_KEYWORDS: tuple[str, ...] = (
    EXCLUDE_KEYWORDS_SONIC + EXCLUDE_KEYWORDS_ACROBATICS + EXCLUDE_KEYWORDS_PAPER
)
# Words that contain an exclude keyword without being related to it ("vegeTABLEs").
_KEYWORD_FALSE_POSITIVES: tuple[str, ...] = ("vegetables",)

DEFAULT_KEEP_ALWAYS = r"peeling_vegetables|washing_dishes|lighter_light_cigarette"
_NAME_RE = re.compile(r"__A\d+(_M)?\.csv$")
_ACT_RE = re.compile(r"__(A\d+)")


# --------------------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------------------


def parse_name(name: str) -> tuple[str, bool, str]:
    """``(stem, is_mirror, performer)`` of a BONES-SEED filename."""
    stem = _NAME_RE.sub("", name)
    act = _ACT_RE.search(name)
    return stem, name.endswith("_M.csv"), act.group(1) if act else ""


def compute_stats(path: str, fps: float = 120.0) -> dict[str, float]:
    """Kinematic stats of one G1 CSV (cols: Frame, root xyz [cm], root euler xyz [deg], 29 dof)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    t = data[:, 1:4] * 0.01
    euler = np.deg2rad(data[:, 4:7])
    q = data[:, 7:]
    z = t[:, 2]
    # World-z component of the body up axis for extrinsic-xyz Euler: R[2, 2] = cos(rx) cos(ry).
    up = np.cos(euler[:, 0]) * np.cos(euler[:, 1])
    qd = np.abs(np.diff(q, axis=0)) * fps
    speed = np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1) * fps
    vz = np.abs(np.diff(z)) * fps
    return {
        "n_frames": float(len(data)),
        "dur": len(data) / fps,
        "z_min": float(z.min()),
        "z_p5": float(np.percentile(z, 5)),
        "z_med": float(np.median(z)),
        "z_max": float(z.max()),
        "z_range": float(z.max() - z.min()),
        "up_min": float(up.min()),
        "up_p5": float(np.percentile(up, 5)),
        "speed_max": float(speed.max()),
        "vz_p99": float(np.percentile(vz, 99)),
        "qd_p99": float(np.percentile(qd, 99)),
        "qd_max": float(qd.max()),
    }


def _stats_task(args: tuple[str, float]) -> dict[str, float]:
    return compute_stats(*args)


def collect_stats(
    files: list[Path], root: Path, workers: int, fps: float, cache: Path | None
) -> list[Record]:
    """Per-file records (``path`` relative to ``root`` + stats), reusing ``cache`` if present."""
    if cache is not None and cache.exists():
        with open(cache, newline="") as handle:
            rows = {r["path"]: r for r in csv.DictReader(handle)}
        wanted = {str(f.relative_to(root)) for f in files}
        if wanted <= rows.keys():
            print(f"[select] stats cache hit: {cache} ({len(wanted)} clips)")
            return [
                {"path": p, **{k: float(rows[p][k]) for k in STAT_FIELDS}} for p in sorted(wanted)
            ]
        print("[select] stats cache is missing clips; recomputing")
    rels = [str(f.relative_to(root)) for f in files]
    jobs = [(str(root / r), fps) for r in rels]
    print(f"[select] computing stats for {len(jobs)} clips with {workers} workers")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            stats = list(pool.map(_stats_task, jobs, chunksize=200))
    else:
        stats = [_stats_task(j) for j in jobs]
    records = [{"path": r, **s} for r, s in zip(rels, stats, strict=True)]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", *STAT_FIELDS])
            writer.writeheader()
            writer.writerows(
                {k: repr(r[k]) if k != "path" else r[k] for k in ["path", *STAT_FIELDS]}
                for r in records
            )
        print(f"[select] wrote stats cache -> {cache}")
    return records


# --------------------------------------------------------------------------------------
# Selection logic (pure functions of name + stats)
# --------------------------------------------------------------------------------------


def motion_group(s: str) -> str:
    """Gating group of a lower-cased stem: ``dance`` > ``jump`` > ``kneel`` > ``other``."""
    group = "other"
    if _KNEEL_RE.search(s):
        group = "kneel"
    if _JUMP_RE.search(s) or _JUMP_EXTRA_RE.search(s):
        group = "jump"
    if _DANCE_RE.search(s):
        group = "dance"
    return group


def drop_reason(s: str, st: dict[str, float], group: str) -> str:
    """First rule that rejects the clip, or ``""`` when it passes the filter."""
    if group == "other":
        for reason, pat in _GENERAL_NAME_RULES:
            if pat.search(s):
                return reason
        for reason, bad in _GENERAL_GATES:
            if bad(st):
                return reason
        return ""
    g = _GROUP_GATES[group]
    if group == "jump":
        if _JUMP_BAD_RE.search(s) or _JUMP_BAD_EXTRA_RE.search(s):
            return f"{group}/name:hard-excluded"
    elif _HARD_BAD_RE.search(s):
        return f"{group}/name:hard-excluded"
    z_range_max = _FROG_Z_RANGE if _FROG_RE.search(s) else g["zr"]
    checks = (
        ("kin:duration", st["dur"] < g["dur"][0] or st["dur"] > g["dur"][1]),
        ("kin:pelvis low", st["z_p5"] < g["z_p5"]),
        ("kin:pelvis high/range", st["z_max"] > g["zmax"] or st["z_range"] > z_range_max),
        ("kin:torso tilt", st["up_p5"] < g["up"]),
        ("kin:root speed", st["speed_max"] > g["sp"]),
        ("kin:vertical speed", st["vz_p99"] > g["vz"]),
        ("kin:joint vel p99", st["qd_p99"] > g["qd99"]),
        ("kin:joint vel max", st["qd_max"] > g["qdmax"]),
    )
    for reason, bad in checks:
        if bad:
            return f"{group}/{reason}"
    return ""


def category(s: str) -> str:
    for label, pattern in _CATEGORIES:
        if re.search(pattern, s):
            return label
    return _EMOTE


def excluded_keyword(s: str) -> str:
    """First ``EXCLUDE_KEYWORDS`` entry contained in the lower-cased stem, or ``""``."""
    for word in _KEYWORD_FALSE_POSITIVES:
        s = s.replace(word, "")
    return next((k for k in EXCLUDE_KEYWORDS if k in s), "")


def classify(records: list[Record], keep_always: str = DEFAULT_KEEP_ALWAYS) -> None:
    """Annotate records in place: stem/mirror/performer/group/category/reason/passed/is_emote."""
    always = re.compile(keep_always) if keep_always else None
    for r in records:
        name = os.path.basename(r["path"])
        stem, mirror, act = parse_name(name)
        s = stem.lower()
        gate_group = motion_group(s)
        keyword = excluded_keyword(s)
        reason = f"keyword:{keyword}" if keyword else drop_reason(s, r, gate_group)
        cat = category(s)
        # dancing_routine is judged by the general gates above (its name has no "dance"), but
        # is a dance for the emote / reporting logic below.
        group = "dance" if _DANCING_ROUTINE_RE.search(s) else gate_group
        if group == "dance" and gate_group == "other":
            cat = "dance_other"
        elif group in ("jump", "kneel"):
            cat = group  # report label only; emote removal only ever applies to group "other"
        r.update(
            name=name,
            stem=stem,
            mirror=mirror,
            act=act,
            group=group,
            category=cat,
            reason=reason,
            passed=reason == "",
            is_emote=group == "other" and cat == _EMOTE and not _LOCOMOTION_RE.search(s),
            always=bool(always and always.search(s)),
        )


def _spread(items: list[Record], n: int) -> list[Record]:
    """``n`` items evenly spread over ``items`` (all of them when ``len(items) <= n``)."""
    if n >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def select(records: list[Record], max_per_motion: int) -> list[Record]:
    """Final clips: passed & non-emote, capped per motion with mirrors paired, + keep_always."""
    by_stem: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r["passed"] and not r["is_emote"]:
            by_stem[r["stem"]].append(r)
    chosen: dict[str, Record] = {}
    for items in by_stem.values():
        items.sort(key=lambda r: (r["act"], r["name"]))
        if max_per_motion <= 0 or len(items) <= max_per_motion:
            picked = items
        else:
            originals = [r for r in items if not r["mirror"]]
            picks = _spread(originals, max_per_motion // 2)
            performers = {r["act"] for r in picks}
            picked = picks + [r for r in items if r["mirror"] and r["act"] in performers]
        chosen.update((r["name"], r) for r in picked)
    for r in records:  # explicitly requested motions bypass the emote filter and the cap
        if r["passed"] and r["always"]:
            chosen[r["name"]] = r
    return sorted(chosen.values(), key=lambda r: r["name"])


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------


def write_outputs(
    records: list[Record], chosen: list[Record], root: Path, out: Path, max_per_motion: int
) -> str:
    out.mkdir(parents=True, exist_ok=True)
    chosen_names = {r["name"] for r in chosen}
    (out / "selected.txt").write_text("".join(f"{root / r['path']}\n" for r in chosen))
    fields = [
        "path",
        "name",
        "stem",
        "mirror",
        "performer",
        "group",
        "category",
        "decision",
        "reason",
        *STAT_FIELDS,
    ]
    with open(out / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            if r["name"] in chosen_names:
                decision, reason = "keep", ""
            elif not r["passed"]:
                decision, reason = "drop", r["reason"]
            elif r["is_emote"]:
                decision, reason = "drop", "emote"
            else:
                decision, reason = "drop", f"cap>{max_per_motion}/motion"
            writer.writerow(
                {
                    "path": r["path"],
                    "name": r["name"],
                    "stem": r["stem"],
                    "mirror": int(r["mirror"]),
                    "performer": r["act"],
                    "group": r["group"],
                    "category": r["category"],
                    "decision": decision,
                    "reason": reason,
                    **{k: f"{r[k]:.6g}" for k in STAT_FIELDS},
                }
            )
    drops = Counter(r["reason"] for r in records if not r["passed"])
    n_pass = sum(r["passed"] for r in records)
    n_emote = sum(r["passed"] and r["is_emote"] and not r["always"] for r in records)
    lines = [
        f"input clips: {len(records)}  (mirrors: {sum(r['mirror'] for r in records)})",
        "\nfilter drops (first-hit rule):",
    ]
    lines += [f"  {n:7d}  {why}" for why, n in drops.most_common()]
    lines += [
        f"passed filter: {n_pass}",
        f"removed as emote: {n_emote}",
        f"selected: {len(chosen)}  (originals {sum(not r['mirror'] for r in chosen)}, "
        f"mirrors {sum(r['mirror'] for r in chosen)}), "
        f"{sum(r['dur'] for r in chosen) / 3600:.1f} h, "
        f"{len({r['stem'] for r in chosen})} motions, "
        f"{len({r['act'] for r in chosen})} performers",
        "\nselected by group:",
    ]
    by_group = Counter(r["group"] for r in chosen)
    lines += [f"  {g:8s} {n:7d}" for g, n in by_group.most_common()]
    lines.append("selected by category:")
    lines += [f"  {c:24s} {n:7d}" for c, n in Counter(r["category"] for r in chosen).most_common()]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    return text


def build_link_dir(chosen: list[Record], root: Path, link_dir: Path) -> None:
    if link_dir.exists() and any(link_dir.iterdir()):
        raise SystemExit(f"--link_dir {link_dir} is not empty; refusing to overwrite")
    names = [r["name"] for r in chosen]
    if len(set(names)) != len(names):
        raise SystemExit("selected filenames are not unique; cannot build a flat link dir")
    link_dir.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        (link_dir / r["name"]).symlink_to((root / r["path"]).resolve())
    print(f"[select] linked {len(chosen)} clips -> {link_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input", required=True, help="G1 BONES-SEED CSV directory (searched recursively)"
    )
    p.add_argument(
        "--output_dir", required=True, help="writes selected.txt / manifest.csv / summary.txt"
    )
    p.add_argument("--link_dir", default="", help="also build a flat symlink dir of the selection")
    p.add_argument(
        "--stats_cache", default="", help="CSV cache of per-clip stats (created if absent)"
    )
    p.add_argument("--workers", type=int, default=8, help="processes for the stats pass")
    p.add_argument(
        "--max_per_motion",
        type=int,
        default=500,
        help="max clips per motion (filename stem) incl. mirrors; 0 = no cap",
    )
    p.add_argument(
        "--keep_always",
        default=DEFAULT_KEEP_ALWAYS,
        help="regex on the lower-cased stem: matching clips that pass the filter bypass "
        "the emote removal and the cap ('' = none)",
    )
    p.add_argument("--input_fps", type=float, default=120.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input).expanduser().resolve()
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files under {root}")
    cache = Path(args.stats_cache).expanduser() if args.stats_cache else None
    records = collect_stats(files, root, args.workers, args.input_fps, cache)
    classify(records, args.keep_always)
    chosen = select(records, args.max_per_motion)
    print(
        write_outputs(
            records, chosen, root, Path(args.output_dir).expanduser(), args.max_per_motion
        )
    )
    if args.link_dir:
        build_link_dir(chosen, root, Path(args.link_dir).expanduser())


if __name__ == "__main__":
    sys.exit(main())
