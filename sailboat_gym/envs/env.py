import gymnasium as gym
import numpy as np
from typing import Any, Tuple, Callable

from ..types import GymObservation, GymAction, Observation, Action, ResetInfo
from ..utils import ProfilingMeta
from ..abstracts import AbcRender


class SailboatEnv(gym.Env, metaclass=ProfilingMeta):
    action_space = GymAction
    observation_space = GymObservation

    def reset(self, **kwargs) -> Tuple[Observation, ResetInfo]:
        super().reset(**kwargs)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Any]:
        super().step(action)

    def render(self, draw_extra_fct: Callable[[AbcRender, np.ndarray, Observation], None] = None) -> np.ndarray:
        super().render()

    def close(self) -> None:
        super().close()
