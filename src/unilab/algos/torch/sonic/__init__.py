"""SONIC G1 policy integration for UniLab (V2 finetune on Intel XPU).

Public API:
  - ``SonicG1Core``      : encoder(+FSQ)+decoder core network (loads sonic last.pt).
  - ``SonicG1ActorModel``: RSL-RL actor wrapper (plug via class_name in the PPO conf).
  - ``SonicCriticModel`` : stock MLP critic + optional warm-load of last.pt's official
    critic (value_state_dict) — plug via class_name; needs the 1645-d critic obs.
"""
from .core import FSQFallback, SonicG1Core, load_critic_from_last_pt, load_g1_from_last_pt, make_fsq
from .critic import SonicCriticModel
from .models import SonicG1ActorModel, SonicH2ActorModel

__all__ = [
    "SonicG1Core",
    "SonicG1ActorModel",
    "SonicH2ActorModel",
    "SonicCriticModel",
    "load_g1_from_last_pt",
    "load_critic_from_last_pt",
    "make_fsq",
    "FSQFallback",
]
