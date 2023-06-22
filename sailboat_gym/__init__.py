from gymnasium.envs.registration import register

from .envs import *
from .renderers import *
from .types import *
from .utils import *
from .abstracts import *

register(
    id='SailboatLSAEnv-v0',
    entry_point='sailboat_gym.envs:SailboatLSAEnv',
    max_episode_steps=60 * SailboatLSAEnv.SIM_RATE,  # 1 minute
)

__all__ = [
    'SailboatLSAEnv',
    'CV2DRenderer',
    'Observation',
    'Action',
    'GymObservation',
    'GymAction'
]

__version__ = '1.0.13'
