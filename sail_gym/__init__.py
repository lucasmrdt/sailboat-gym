from gymnasium.envs.registration import register

from .envs import *
from .renderers import *
from .types import *
from .utils import *

register(
    id='SailboatLSAEnv-v0',
    entry_point='sail_gym.envs:SailboatLSAEnv',
    max_episode_steps=60 * SailboatLSAEnv.SIM_RATE,  # 1 minute
)
