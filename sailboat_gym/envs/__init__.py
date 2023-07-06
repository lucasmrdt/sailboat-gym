from gymnasium.envs.registration import register

from .sailboat_lsa import SailboatLSAEnv
from .env import *

env_by_name = {
    'SailboatLSAEnv-v0': SailboatLSAEnv,
}

register(
    id='SailboatLSAEnv-v0',
    entry_point='sailboat_gym.envs:SailboatLSAEnv',
)
