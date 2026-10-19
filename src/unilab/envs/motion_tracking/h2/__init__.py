"""Motion tracking environments for Unitree H2 (29-DOF, head welded like Unitree G1)."""

from .tracking_sonic import H2SonicMotionTrackingCfg, H2SonicMotionTrackingEnv

__all__ = [
    "H2SonicMotionTrackingCfg",
    "H2SonicMotionTrackingEnv",
]
