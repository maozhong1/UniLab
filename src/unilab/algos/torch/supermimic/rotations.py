"""Batched, differentiable quaternion helpers (torch, **wxyz**, arbitrary leading dims).

Same conventions as mjlab / Isaac Lab ``math`` (which SuperMimic re-exports), plus the
operations the world model needs: rotation-vector exp/log, 6D rotation features and heading
(yaw) frames.
"""

from __future__ import annotations

import math

import torch

_EPS = 1.0e-8


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return quat_conjugate(q) / (q * q).sum(-1, keepdim=True).clamp_min(_EPS)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle.unsqueeze(-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so that w >= 0 (removes the double-cover ambiguity)."""
    return torch.where(q[..., :1] < 0, -q, q)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """Exponential map; a Taylor branch keeps gradients finite at v = 0."""
    angle_sq = (v * v).sum(-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1e-30))
    small = angle < 1e-4
    k = torch.where(small, 0.5 - angle_sq / 48.0, torch.sin(0.5 * angle) / angle)
    w = torch.where(small, 1.0 - angle_sq / 8.0, torch.cos(0.5 * angle))
    return torch.cat([w, v * k], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Logarithmic map to a rotation vector in (-pi, pi]; finite gradient at the identity."""
    q = quat_canonical(q)
    w = q[..., :1]
    xyz = q[..., 1:]
    s = (xyz * xyz).sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    angle = 2.0 * torch.atan2(s, w)
    small = s < 1e-6
    k = torch.where(
        small, 2.0 + (angle * angle) / 12.0, angle / torch.where(small, torch.ones_like(s), s)
    )
    return xyz * k


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, flattened (..., 6)."""
    return matrix_from_quat(q)[..., :, :2].reshape(*q.shape[:-1], 6)


def heading_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Pure-yaw (heading) quaternion of q."""
    return quat_from_yaw(heading_yaw(q))


heading_quat = yaw_quat


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def projected_gravity(q_body_w: torch.Tensor) -> torch.Tensor:
    """Gravity direction (0, 0, -1) expressed in the body frame."""
    g = torch.tensor([0.0, 0.0, -1.0], dtype=q_body_w.dtype, device=q_body_w.device)
    return quat_apply_inverse(q_body_w, g.expand(*q_body_w.shape[:-1], 3))
