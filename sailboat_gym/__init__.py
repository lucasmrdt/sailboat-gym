from gymnasium.envs.registration import register

from .envs import *
from .renderers import *
from .types import *
from .utils import *
from .abstracts import *

NB_STEPS_PER_SECONDS = SailboatLSAEnv.SIM_RATE

register(
    id='SailboatLSAEnv-v0',
    entry_point='sailboat_gym.envs:SailboatLSAEnv',
)

__all__ = [
    'SailboatLSAEnv',
    'CV2DRenderer',
    'Observation',
    'Action',
    'GymObservation',
    'GymAction'
]

__version__ = '1.0.15'
