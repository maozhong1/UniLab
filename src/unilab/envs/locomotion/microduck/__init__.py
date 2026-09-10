"""microduck locomotion tasks (ported from microduck_rl)."""

from unilab.envs.locomotion.microduck.actuator import (
    BamActuatorConfig,
    MicroduckBamActuator,
)
from unilab.envs.locomotion.microduck.base import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    NUM_COMMAND,
    NUM_SERVOS,
    SERVO_NAMES,
    MicroduckBaseCfg,
    MicroduckBaseEnv,
    MicroduckCommandRanges,
    MicroduckControlConfig,
    MicroduckDomainRandConfig,
    MicroduckDRProvider,
    MicroduckNoiseConfig,
    MicroduckSensor,
)
from unilab.envs.locomotion.microduck.velocity import (
    MicroduckVelocityEnv,
    MicroduckVelocityFlatCfg,
    MicroduckVelocityRewardCfg,
)
from unilab.envs.locomotion.microduck.rollers import (
    MicroduckRollersCfg,
    MicroduckRollersEnv,
    MicroduckRollersRewardCfg,
)

__all__ = [
    "ACTOR_OBS_DIM",
    "CRITIC_OBS_DIM",
    "NUM_COMMAND",
    "NUM_SERVOS",
    "SERVO_NAMES",
    "BamActuatorConfig",
    "MicroduckBamActuator",
    "MicroduckBaseCfg",
    "MicroduckBaseEnv",
    "MicroduckCommandRanges",
    "MicroduckControlConfig",
    "MicroduckDomainRandConfig",
    "MicroduckDRProvider",
    "MicroduckNoiseConfig",
    "MicroduckSensor",
    "MicroduckVelocityEnv",
    "MicroduckVelocityFlatCfg",
    "MicroduckVelocityRewardCfg",
    "MicroduckRollersCfg",
    "MicroduckRollersEnv",
    "MicroduckRollersRewardCfg",
]
