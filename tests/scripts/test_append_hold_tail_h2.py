from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "motion" / "append_hold_tail_h2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("append_hold_tail_h2", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clip(path: Path, foot_ids: list[int], *, swing_at_end: bool) -> None:
    t, nb = 20, 33
    body_pos = np.zeros((t, nb, 3), np.float32)
    body_vel = np.ones((t, nb, 3), np.float32) * 0.5
    body_pos[:, foot_ids, 2] = 0.05
    body_vel[:, foot_ids] = 0.0
    if swing_at_end:
        body_pos[-1, foot_ids[0], 2] = 0.20
    np.savez(path, fps=np.array([50], np.int32), joint_pos=np.random.rand(t, 29).astype(np.float32),
             joint_vel=np.ones((t, 29), np.float32), body_pos_w=body_pos,
             body_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (t, nb, 1)),
             body_lin_vel_w=body_vel, body_ang_vel_w=np.ones((t, nb, 3), np.float32))


def test_hold_appended_only_to_double_support_endings(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    foot_ids = mod._foot_ids(mod.DEFAULT_H2_MODEL)
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _clip(src / "stand.npz", foot_ids, swing_at_end=False)
    _clip(src / "stride.npz", foot_ids, swing_at_end=True)
    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--dst", str(dst), "--hold-s", "0.5"])
    mod.main()

    assert (dst / "stride.npz").is_symlink()
    out, orig = np.load(dst / "stand.npz"), np.load(src / "stand.npz")
    assert out["joint_pos"].shape == (45, 29)  # 20 + 0.5 s * 50 fps
    np.testing.assert_array_equal(out["joint_pos"][20:], np.repeat(orig["joint_pos"][-1:], 25, 0))
    np.testing.assert_array_equal(out["body_pos_w"][20:], np.repeat(orig["body_pos_w"][-1:], 25, 0))
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        assert not out[key][20:].any()
        np.testing.assert_array_equal(out[key][:20], orig[key])
