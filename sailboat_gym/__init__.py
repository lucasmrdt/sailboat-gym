from gymnasium.envs.registration import register

from .envs import *
from .renderers import *
from .types import *
from .utils import *
from .abstracts import *

EPISODE_LENGTH = 60 * SailboatLSAEnv.SIM_RATE  # 60 seconds * 10 steps per second

register(
    id='SailboatLSAEnv-v0',
    entry_point='sailboat_gym.envs:SailboatLSAEnv',
    # max_episode_steps=EPISODE_LENGTH,
)

__all__ = [
    'SailboatLSAEnv',
    'CV2DRenderer',
    'Observation',
    'Action',
    'GymObservation',
    'GymAction'
]

__version__ = '1.0.14'
