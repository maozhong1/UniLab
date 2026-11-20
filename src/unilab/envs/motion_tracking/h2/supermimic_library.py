"""Multi-clip reference library and return-based reset sampler for the H2 SuperMimic task.

NumPy port of SuperMimic's ``MotionLibrary`` / ``MultiClipMotionCommand`` sampling
(``supermimic/envs/mdp/{motion_library,commands}.py``). All clips are concatenated along one
global frame axis; per-clip tables map a global frame to its clip, its last frame and its 1 s
sampling bin.

Goal features of one reference frame (2N+10, N = 29 for H2 -> 68):
    [joint_pos - q0 (N), joint_vel (N), projected gravity of the ref root (3),
     ref root angular velocity in the root frame (3),
     ref root linear velocity in the ref heading frame (3), ref root height (1)]
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unilab.utils.rotation import np_quat_apply_batched, np_yaw_quat

from ..common.motion_loader import MotionLoader

_GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` (..., 3) by the inverse of unit quaternion ``q`` (..., 4, wxyz)."""
    return np_quat_apply_batched(_quat_conj(q), v)


def discover_motion_files(
    motion_files: str | Sequence[str] = (),
    motion_dirs: Sequence[str] = (),
    manifest: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Expand explicit files plus every ``*.npz`` of ``motion_dirs`` (sorted per directory).

    ``manifest`` is a csv with ``name,keep`` columns (SuperMimic ``filter_bones_seed.py`` output);
    only rows with keep == 1 are retained.
    """
    files: list[str] = []
    if isinstance(motion_files, str):
        motion_files = [motion_files] if motion_files else []
    files.extend(str(Path(f).expanduser()) for f in motion_files)
    for d in motion_dirs:
        path = Path(d).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"motion directory '{path}' does not exist")
        files.extend(str(f) for f in sorted(path.glob("*.npz")))
    if manifest:
        with open(Path(manifest).expanduser()) as f:
            keep = {
                row["name"]
                for row in csv.DictReader(f)
                if row.get("keep", "1") in ("1", "True", "true")
            }
        files = [f for f in files if Path(f).stem in keep]
    if limit is not None and limit > 0:
        files = files[:limit]
    if not files:
        raise ValueError("no motion files: set motion_file and/or motion_dirs")
    return files


@dataclass
class RefFrames:
    """Reference quantities of a batch of global frames."""

    root_pos_w: np.ndarray  # (n, 3)
    root_quat_w: np.ndarray  # (n, 4) wxyz
    root_lin_vel_w: np.ndarray  # (n, 3)
    root_ang_vel_w: np.ndarray  # (n, 3) world frame
    joint_pos: np.ndarray  # (n, N)
    joint_vel: np.ndarray  # (n, N)
    body_pos_w: np.ndarray  # (n, B, 3) tracked bodies
    body_quat_w: np.ndarray  # (n, B, 4)
    body_lin_vel_w: np.ndarray  # (n, B, 3)
    body_ang_vel_w: np.ndarray  # (n, B, 3)


class SuperMimicMotionLibrary:
    """Concatenated reference clips (tracked bodies only) with clip / bin lookup tables."""

    def __init__(
        self,
        motion_files: Sequence[str],
        motion_body_ids: np.ndarray,
        root_index: int,
        default_joint_pos: np.ndarray,
        bin_frames: int = 50,
    ):
        loader = MotionLoader(list(motion_files), body_indices=np.asarray(motion_body_ids))
        if loader.fps != 50:
            raise ValueError(f"SuperMimic expects 50 fps reference motions, got {loader.fps}")
        self.motion_files = tuple(loader.motion_files)
        self.clip_names = tuple(Path(f).stem for f in self.motion_files)
        self.joint_pos = loader.joint_pos
        self.joint_vel = loader.joint_vel
        self.body_pos_w = loader.body_pos_w
        self.body_quat_w = loader.body_quat_w
        self.body_lin_vel_w = loader.body_lin_vel_w
        self.body_ang_vel_w = loader.body_ang_vel_w
        self.root_index = int(root_index)
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)
        if self.default_joint_pos.shape != (self.joint_pos.shape[1],):
            raise ValueError(
                f"default pose has {self.default_joint_pos.shape} joints, motions have {self.joint_pos.shape[1]}"
            )

        self.num_frames = int(loader.num_frames)
        self.num_clips = int(loader.num_clips)
        self.clip_start = loader.clip_offsets.astype(np.int64)
        self.clip_len = loader.clip_lengths.astype(np.int64)
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = np.repeat(np.arange(self.num_clips), self.clip_len)

        # 1 s bins inside each clip, numbered globally (return-based reset sampling).
        self.bin_frames = int(bin_frames)
        bins_per_clip = (self.clip_len + self.bin_frames - 1) // self.bin_frames
        self.clip_bin_offset = np.cumsum(bins_per_clip) - bins_per_clip
        self.num_bins = int(bins_per_clip.sum())
        bin_clip = np.repeat(np.arange(self.num_clips), bins_per_clip)
        local_bin = np.arange(self.num_bins) - self.clip_bin_offset[bin_clip]
        self.bin_start = self.clip_start[bin_clip] + local_bin * self.bin_frames
        self.bin_end = np.minimum(
            self.bin_start + self.bin_frames, self.clip_start[bin_clip] + self.clip_len[bin_clip]
        )
        self.bin_clip = bin_clip

    # ------------------------------------------------------------------ lookups

    def last_frame_of(self, frame: np.ndarray) -> np.ndarray:
        return self.clip_last[self.frame_clip[frame]]

    def bin_of_frame(self, frame: np.ndarray) -> np.ndarray:
        clip = self.frame_clip[frame]
        return self.clip_bin_offset[clip] + (frame - self.clip_start[clip]) // self.bin_frames

    def ref(self, frame: np.ndarray) -> RefFrames:
        r = self.root_index
        return RefFrames(
            root_pos_w=self.body_pos_w[frame, r],
            root_quat_w=self.body_quat_w[frame, r],
            root_lin_vel_w=self.body_lin_vel_w[frame, r],
            root_ang_vel_w=self.body_ang_vel_w[frame, r],
            joint_pos=self.joint_pos[frame],
            joint_vel=self.joint_vel[frame],
            body_pos_w=self.body_pos_w[frame],
            body_quat_w=self.body_quat_w[frame],
            body_lin_vel_w=self.body_lin_vel_w[frame],
            body_ang_vel_w=self.body_ang_vel_w[frame],
        )

    def ref_states(self) -> np.ndarray:
        """(F, 13+2N) reference states ``[root_pos, root_quat, q, root_lin_vel_w, root_ang_vel_w, qd]``.

        Same layout as SuperMimic's packed library (root angular velocity in the world frame).
        """
        r = self.root_index
        return np.concatenate(
            [
                self.body_pos_w[:, r],
                self.body_quat_w[:, r],
                self.joint_pos,
                self.body_lin_vel_w[:, r],
                self.body_ang_vel_w[:, r],
                self.joint_vel,
            ],
            axis=1,
        ).astype(np.float32)

    def goal_features(self, frame: np.ndarray) -> np.ndarray:
        """(..., 2N+10) goal features of global frames ``frame`` (any shape)."""
        r = self.root_index
        q = self.body_quat_w[frame, r]
        grav = quat_apply_inverse(q, np.broadcast_to(_GRAVITY_W, q.shape[:-1] + (3,)))
        ang_b = quat_apply_inverse(q, self.body_ang_vel_w[frame, r])
        yaw_q = np_yaw_quat(q.reshape(-1, 4)).reshape(q.shape).astype(np.float32)
        lin_h = quat_apply_inverse(yaw_q, self.body_lin_vel_w[frame, r])
        height = self.body_pos_w[frame, r, 2:3]
        return np.concatenate(
            [
                self.joint_pos[frame] - self.default_joint_pos,
                self.joint_vel[frame],
                grav,
                ang_b,
                lin_h,
                height,
            ],
            axis=-1,
        )

    def goal_window(self, frame: np.ndarray, k: int) -> np.ndarray:
        """(n, k*(2N+10)) goal features of frames frame..frame+k-1, clamped at the clip end."""
        idx = np.minimum(frame[:, None] + np.arange(k)[None, :], self.last_frame_of(frame)[:, None])
        feat = self.goal_features(idx)
        return feat.reshape(frame.shape[0], -1).astype(np.float32, copy=False)

    def summary(self) -> str:
        hours = self.num_frames / 50.0 / 3600.0
        return f"SuperMimicMotionLibrary: {self.num_clips} clips, {self.num_frames} frames ({hours:.2f} h @50Hz), {self.num_bins} bins"


@dataclass
class ReturnSamplerCfg:
    """Return-based reset sampling over 1 s bins (SuperMimic plan D14)."""

    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0
    g_max: float = 10.0
    gamma: float = 0.99
    ema_alpha: float = 0.05
    memory_alpha: float = 0.01
    min_remaining_steps: int = 100
    max_logged_steps: int = 1024


class ReturnBasedSampler:
    """Chooses reset frames; bins with a low discounted return-to-go are sampled more often."""

    def __init__(
        self,
        lib: SuperMimicMotionLibrary,
        num_envs: int,
        mode: str,
        max_episode_steps: int,
        cfg: ReturnSamplerCfg,
        rng: np.random.Generator,
    ):
        if mode not in ("return", "uniform", "start"):
            raise ValueError(f"unknown sampling_mode '{mode}' (return | uniform | start)")
        self.lib, self.mode, self.cfg, self.rng = lib, mode, cfg, rng
        self.bin_return = np.full(lib.num_bins, cfg.g_max, dtype=np.float64)
        self.bin_memory = np.full(lib.num_bins, 1.0 / lib.num_bins, dtype=np.float64)
        self.max_steps = min(max_episode_steps + 2, cfg.max_logged_steps)
        self.ep_rewards = np.zeros((num_envs, self.max_steps), dtype=np.float64)
        self.ep_bins = np.zeros((num_envs, self.max_steps), dtype=np.int64)
        self.ep_len = np.zeros(num_envs, dtype=np.int64)
        self.gamma_pow = cfg.gamma ** np.arange(self.max_steps, dtype=np.float64)
        self.entropy = 1.0

    def log_step(self, reward: np.ndarray, frame: np.ndarray) -> None:
        """Append every env's reward of this step and the bin of its target frame."""
        pos = np.minimum(self.ep_len, self.max_steps - 1)
        rows = np.arange(self.ep_len.shape[0])
        self.ep_rewards[rows, pos] = reward
        self.ep_bins[rows, pos] = self.lib.bin_of_frame(frame)
        self.ep_len = np.minimum(self.ep_len + 1, self.max_steps)

    def finalize(self, env_ids: np.ndarray, terminated: np.ndarray, clip_end: np.ndarray) -> None:
        """Fold the finished episodes' discounted returns-to-go into the per-bin EMA."""
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        if lens.sum() == 0:
            return
        T = int(lens.max())
        steps = np.arange(T)[None, :]
        tmask = steps < lens[:, None]
        r = self.ep_rewards[env_ids, :T] * tmask
        disc = r * self.gamma_pow[None, :T]
        rtg = np.flip(np.cumsum(np.flip(disc, 1), 1), 1) / self.gamma_pow[None, :T]
        # Truncated (time-out) episodes: drop the steps whose return-to-go is cut short.
        truncated = ~terminated & ~clip_end
        if truncated.any():
            remaining = lens[:, None] - steps
            tmask = tmask & ~(truncated[:, None] & (remaining < cfg.min_remaining_steps))
        b = self.ep_bins[env_ids, :T][tmask]
        g = rtg[tmask]
        self.ep_len[env_ids] = 0
        if b.size == 0:
            return
        sums = np.bincount(b, weights=g, minlength=self.lib.num_bins)
        cnts = np.bincount(b, minlength=self.lib.num_bins)
        touched = cnts > 0
        self.bin_return[touched] = (1 - cfg.ema_alpha) * self.bin_return[
            touched
        ] + cfg.ema_alpha * sums[touched] / cnts[touched]

    def sample(self, n: int) -> np.ndarray:
        lib, cfg = self.lib, self.cfg
        if self.mode == "start":
            return lib.clip_start[self.rng.integers(0, lib.num_clips, n)]
        if self.mode == "uniform":
            f = self.rng.integers(0, lib.num_frames, n)
            return np.maximum(
                np.minimum(f, lib.last_frame_of(f) - 2), lib.clip_start[lib.frame_clip[f]]
            )
        logits = -cfg.softmax_temp * self.bin_return / cfg.g_max
        hard = np.exp(logits - logits.max())
        hard /= hard.sum()
        self.bin_memory = (1 - cfg.memory_alpha) * self.bin_memory + cfg.memory_alpha * hard
        p = (
            cfg.uniform_ratio / lib.num_bins
            + cfg.hard_ratio * hard
            + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        p /= p.sum()
        b = self.rng.choice(lib.num_bins, size=n, p=p)
        u = self.rng.random(n)
        f = lib.bin_start[b] + (u * (lib.bin_end[b] - lib.bin_start[b])).astype(np.int64)
        f = np.maximum(np.minimum(f, lib.last_frame_of(f) - 2), lib.clip_start[lib.frame_clip[f]])
        self.entropy = float(-(p * np.log(p + 1e-12)).sum() / math.log(max(lib.num_bins, 2)))
        return f
