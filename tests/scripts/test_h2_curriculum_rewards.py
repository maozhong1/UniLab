"""H2 curriculum reward overrides reach Hydra without changing earlier stages."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from unilab.envs.motion_tracking.h2.tracking_sonic import SonicRewardConfig
from unilab.training.reward import extract_reward_config

ROOT_DIR = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("stage", "scale_overrides", "expected"),
    [
        ("1", {}, (0.0, 0.0)),
        ("2.5", {}, (0.0, 0.0)),
        ("2.8", {}, (0.0, 0.0)),
        ("2.85", {}, (2.0, -0.1)),
        (
            "2.85",
            {"MOTION_LOCAL_POINTS_SCALE": "0.0", "UNDESIRED_CONTACTS_SCALE": "0.0"},
            (0.0, 0.0),
        ),
        (
            "2.85",
            {"MOTION_LOCAL_POINTS_SCALE": "1.5", "UNDESIRED_CONTACTS_SCALE": "-0.2"},
            (1.5, -0.2),
        ),
    ],
)
def test_h2_curriculum_reward_compose(tmp_path, stage, scale_overrides, expected):
    source = tmp_path / "source"
    manipulation = tmp_path / "manipulation"
    source.mkdir()
    manipulation.mkdir()
    (source / "base.npz").touch()
    (manipulation / "manipulation.npz").touch()
    work = tmp_path / "work"
    filtered = work / f"h2new_stage{stage.replace('.', '_')}"
    filtered.mkdir(parents=True)
    (filtered / "base.npz").touch()
    run_name = "resume_run"
    run_dir = tmp_path / "logs" / "rsl_rl_ppo" / "H2SonicMotionTracking" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "model_5100.pt").touch()
    (run_dir / "model_5200.pt").touch()
    captured_args = tmp_path / "train_args"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'if [[ "$4" == "scripts/train_rsl_rl.py" ]]; then\n'
        '  shift 4\n  printf "%s\\n" "$@" > "$CAPTURED_ARGS"\n'
        'elif [[ "$4" != "scripts/motion/filter_calm_h2.py" ]]; then\n'
        '  echo "unexpected command: $*" >&2\n  exit 1\nfi\n'
    )
    uv.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.defpath}",
        "UNILAB": str(tmp_path),
        "SOURCE_NPZ_DIR": str(source),
        "MANIPULATION_NPZ_DIR": str(manipulation),
        "WORK_DIR": str(work),
        "CURRICULUM_STAGE": stage,
        "LOAD_RUN": run_name,
        "CHECKPOINT": "5100",
        "HOLD_TAIL_S": "0",
        "FOCUS_PATTERN": "",
        "EXTRA_FOCUS": "",
        "DRY_RUN": "1",
        "CAPTURED_ARGS": str(captured_args),
        **scale_overrides,
    }
    result = subprocess.run(
        ["bash", str(ROOT_DIR / "scripts/sonic/full_train_h2_new.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = captured_args.read_text().splitlines()
    assert args[-2:] == ["--cfg", "job"]
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=args[:-2])
    assert cfg.reward.scales.motion_local_points == expected[0]
    assert cfg.reward.scales.undesired_contacts == expected[1]
    reward_config = SonicRewardConfig(**extract_reward_config(cfg)["reward_config"])
    assert reward_config.scales["motion_local_points"] == expected[0]
    assert reward_config.scales["undesired_contacts"] == expected[1]
    assert reward_config.std_local_points == 0.1
    assert cfg.reward.scales.feet_slide == -1.0
    assert cfg.algo.resume is True
    assert cfg.algo.load_run == run_name
    assert cfg.algo.checkpoint == 5100
    if stage == "2.85":
        assert cfg.reward.scales.upper_body_joint_pos == 2.0
        assert cfg.reward.std_upper_body_joint_pos == 0.35
        assert cfg.reward.anti_shake_threshold == 4.0
        assert cfg.algo.max_iterations == 1000
        assert cfg.algo.algorithm.critic_warmup_iters == 30


def test_h2_full_train_reward_defaults_remain_off():
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=h2_motion_tracking/sonic_full_train"])
    assert cfg.reward.scales.motion_local_points == 0.0
    assert cfg.reward.scales.undesired_contacts == 0.0
