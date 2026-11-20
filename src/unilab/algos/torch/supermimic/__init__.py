"""SuperMimic: one learned world model + BPTT for humanoid whole-body motion tracking.

Port of the SuperMimic reference implementation (mjlab / MuJoCo Warp) to UniLab. Real
transitions come from a UniLab ``NpEnv`` (``H2SuperMimic``); the world model, the analytic
FK reward / observation heads, the policy and the critic are pure torch on the learner device.
"""

from .runner import ScheduleCfg, SuperMimicCfg, SuperMimicRunner, build_models

__all__ = ["ScheduleCfg", "SuperMimicCfg", "SuperMimicRunner", "build_models"]
