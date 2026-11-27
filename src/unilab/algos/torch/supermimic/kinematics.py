"""Differentiable batched forward kinematics for a free-floating humanoid (torch).

Built from a compiled ``mujoco.MjModel`` (G1, GR2, ...) so the tree, joint axes,
anchors and ``qpos0`` are exactly MuJoCo's. Reproduces, for every body:

* ``xpos`` / ``xquat``                      (link frame origin pose, world)
* link-frame-origin linear velocity, world  (== mjlab ``body_link_lin_vel_w``)
* angular velocity, world                   (== mjlab ``body_link_ang_vel_w``)

given ``qpos`` (7+N: root pos, root quat wxyz, N hinges) and ``qvel`` (6+N: root
linear velocity in world, root angular velocity in **body** frame, N hinge rates),
i.e. MuJoCo's free-joint conventions.

Quaternion math is element-wise (no 3x3 matmuls) so results are unaffected by TF32.
Position FK follows ``mj_kinematics`` (hinge rotates about its anchor by
``qpos - qpos0``); velocities are propagated down the tree:

    w_b = w_parent + axis_w * qdot
    v_b = v_parent + w_parent x (p_b - p_parent) + (axis_w * qdot) x (p_b - anchor_w)

which equals MuJoCo's ``cvel`` transported to the body origin (``mj_objectVelocity``
with ``flg_local=0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from .rotations import quat_apply, quat_from_angle_axis, quat_mul, quat_normalize


@dataclass
class _BodyLink:
    body_id: int  # MuJoCo body id (1..nbody-1)
    parent_index: int  # index into our body list (0..nb-1), -1 for the root
    joint_type: int  # mjJNT_FREE / mjJNT_HINGE / -1 (fixed)
    qpos_adr: int
    dof_adr: int


class HumanoidKinematics:
    """Batched differentiable FK for a MuJoCo single-tree humanoid (free root + hinges)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.nbody = model.nbody - 1  # exclude world
        self.nq = model.nq
        self.nv = model.nv
        self.body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)
        ]
        self.joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
        ]

        links: list[_BodyLink] = []
        for b in range(1, model.nbody):
            parent = int(model.body_parentid[b])
            parent_index = -1 if parent == 0 else parent - 1
            assert parent < b, "MuJoCo bodies are topologically ordered"
            njnt = int(model.body_jntnum[b])
            if njnt == 0:
                links.append(_BodyLink(b, parent_index, -1, -1, -1))
                continue
            assert njnt == 1, f"body {b} has {njnt} joints; only 0/1 supported"
            j = int(model.body_jntadr[b])
            jt = int(model.jnt_type[j])
            assert jt in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE), (
                f"joint type {jt}"
            )
            links.append(
                _BodyLink(b, parent_index, jt, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
            )
        assert links[0].joint_type == mujoco.mjtJoint.mjJNT_FREE and links[0].parent_index == -1
        self.links = links
        # Plain Python flags for forward(): comparing against the pybind ``mjtJoint`` enum inside
        # the loop makes ``torch.compile`` graph-break once per body.
        self._is_hinge = [l.joint_type == mujoco.mjtJoint.mjJNT_HINGE for l in links]

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)  # noqa: E731
        self.body_pos = t(model.body_pos[1:])  # (nb,3)
        self.body_quat = t(model.body_quat[1:])  # (nb,4) wxyz
        jnt_pos = np.zeros((self.nbody, 3))
        jnt_axis = np.zeros((self.nbody, 3))
        qpos0 = np.zeros(self.nbody)
        for i, l in enumerate(links):
            if l.joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                j = int(model.body_jntadr[l.body_id])
                jnt_pos[i] = model.jnt_pos[j]
                jnt_axis[i] = model.jnt_axis[j]
                qpos0[i] = model.qpos0[l.qpos_adr]
        self.jnt_pos = t(jnt_pos)
        self.jnt_axis = t(jnt_axis)
        self.qpos0 = t(qpos0)
        # Hinge bookkeeping in body order (skip root / fixed bodies).
        self.hinge_body_idx = [
            i for i, l in enumerate(links) if l.joint_type == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.hinge_qpos_adr = torch.tensor(
            [links[i].qpos_adr for i in self.hinge_body_idx], device=self.device
        )
        self.hinge_dof_adr = torch.tensor(
            [links[i].dof_adr for i in self.hinge_body_idx], device=self.device
        )
        self._build_levels()

    def _build_levels(self) -> None:
        """Group bodies by tree depth. Bodies of one depth are independent, so FK runs one batched
        step per level (~8) instead of one per body (33): far fewer kernels, which is what bounds
        the SuperMimic world-model / imagination updates on XPU."""
        nb = self.nbody
        depth = [0] * nb
        for i in range(1, nb):
            depth[i] = depth[self.links[i].parent_index] + 1
        levels = [[i for i in range(1, nb) if depth[i] == d] for d in range(1, max(depth) + 1)]
        order = [0] + [i for lv in levels for i in lv]  # body ids in storage order
        slot = {b: k for k, b in enumerate(order)}  # body id -> storage slot
        self._inv_order = torch.tensor([slot[b] for b in range(nb)], device=self.device)
        dev = self.device
        self._levels = []
        for lv in levels:
            idx = torch.tensor(lv, device=dev)
            hinge = torch.tensor(
                [float(self._is_hinge[i]) for i in lv], dtype=self.dtype, device=dev
            )
            self._levels.append(
                dict(
                    parent=torch.tensor([slot[self.links[i].parent_index] for i in lv], device=dev),
                    body_pos=self.body_pos[idx],
                    body_quat=self.body_quat[idx],
                    jnt_pos=self.jnt_pos[idx],
                    jnt_axis=self.jnt_axis[idx],
                    qpos0=self.qpos0[idx],
                    hinge=hinge,
                    qadr=torch.tensor([max(self.links[i].qpos_adr, 0) for i in lv], device=dev),
                    dadr=torch.tensor([max(self.links[i].dof_adr, 0) for i in lv], device=dev),
                )
            )

    def body_index(self, name: str) -> int:
        return self.body_names.index(name)

    def body_indices(self, names) -> torch.Tensor:
        return torch.tensor([self.body_names.index(n) for n in names], device=self.device)

    # ------------------------------------------------------------------ core

    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None):
        """FK for a batch.

        Args:
          qpos: (..., nq) MuJoCo generalized positions.
          qvel: (..., nv) MuJoCo generalized velocities (free joint: lin world, ang body).
        Returns:
          dict with 'pos' (..., nb, 3), 'quat' (..., nb, 4) and, if qvel is given,
          'lin_vel' (..., nb, 3), 'ang_vel' (..., nb, 3) in world frame at the link origin.
        """
        compute_vel = qvel is not None

        # Root (free joint) occupies storage slot 0; every level is appended after its parents.
        root_quat = quat_normalize(qpos[..., 3:7])
        P_pos = qpos[..., 0:3].unsqueeze(-2)
        P_quat = root_quat.unsqueeze(-2)
        if compute_vel:
            P_lin = qvel[..., 0:3].unsqueeze(-2)
            P_ang = quat_apply(root_quat, qvel[..., 3:6]).unsqueeze(-2)

        for lv in self._levels:
            par = lv["parent"]
            pq = P_quat.index_select(-2, par)
            pp = P_pos.index_select(-2, par)
            ex = lambda x: x.expand(*pq.shape[:-1], x.shape[-1])  # noqa: E731
            b_pos = pp + quat_apply(pq, ex(lv["body_pos"]))
            b_quat = quat_mul(pq, ex(lv["body_quat"]))
            # Fixed bodies carry zero joint anchor / axis and zero angle, so the hinge formulas
            # below reduce to the plain fixed-offset case for them.
            anchor = b_pos + quat_apply(b_quat, ex(lv["jnt_pos"]))
            axis_w = quat_apply(b_quat, ex(lv["jnt_axis"]))
            angle = (qpos[..., lv["qadr"]] - lv["qpos0"]) * lv["hinge"]
            b_quat = quat_mul(b_quat, quat_from_angle_axis(angle, ex(lv["jnt_axis"])))
            b_pos = anchor - quat_apply(b_quat, ex(lv["jnt_pos"]))
            if compute_vel:
                p_ang = P_ang.index_select(-2, par)
                w_joint = axis_w * (qvel[..., lv["dadr"]] * lv["hinge"]).unsqueeze(-1)
                lin = (
                    P_lin.index_select(-2, par)
                    + torch.cross(p_ang, b_pos - pp, dim=-1)
                    + torch.cross(w_joint, b_pos - anchor, dim=-1)
                )
                P_lin = torch.cat([P_lin, lin], dim=-2)
                P_ang = torch.cat([P_ang, p_ang + w_joint], dim=-2)
            P_pos = torch.cat([P_pos, b_pos], dim=-2)
            P_quat = torch.cat([P_quat, b_quat], dim=-2)

        inv = self._inv_order
        out = {"pos": P_pos.index_select(-2, inv), "quat": P_quat.index_select(-2, inv)}
        if compute_vel:
            out["lin_vel"] = P_lin.index_select(-2, inv)
            out["ang_vel"] = P_ang.index_select(-2, inv)
        return out

    __call__ = forward
