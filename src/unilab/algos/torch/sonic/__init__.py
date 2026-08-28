"""SONIC G1 policy integration for UniLab (V2 finetune on Intel XPU).

Public API:
  - ``SonicG1Core``      : encoder(+FSQ)+decoder core network (loads sonic last.pt).
  - ``SonicG1ActorModel``: RSL-RL actor wrapper (plug via class_name in the PPO conf).

The critic stays a stock ``rsl_rl.models.MLPModel`` (configured in YAML); no custom
critic is needed because the value net should not use the quantized FSQ backbone.
"""
from .core import FSQFallback, SonicG1Core, load_g1_from_last_pt, make_fsq
from .models import SonicG1ActorModel

__all__ = [
    "SonicG1Core",
    "SonicG1ActorModel",
    "load_g1_from_last_pt",
    "make_fsq",
    "FSQFallback",
]
