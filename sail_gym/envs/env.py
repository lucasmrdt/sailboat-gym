import gymnasium as gym
import numpy as np
from typing import Any, Tuple

from ..types import GymObservation, GymAction, Observation, Action
from ..utils import ProfilingMeta


class SailboatEnv(gym.Env, metaclass=ProfilingMeta):
    action_space = GymAction
    observation_space = GymObservation

    def reset(self, **kwargs) -> Observation:
        super().reset(**kwargs)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Any]:
        super().step(action)

    def render(self) -> np.ndarray:
        super().render()

    def close(self) -> None:
        super().close()
