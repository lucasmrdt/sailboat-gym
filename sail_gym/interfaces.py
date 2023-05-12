import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List

from .types import Observation
from .utils import ProfilingMeta


class ABCProfilingMeta(ABCMeta, ProfilingMeta):
    pass


class IRenderer(metaclass=ABCProfilingMeta):
    @abstractmethod
    def get_render_mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_render_modes(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def setup(self, min_position: np.ndarray, max_position: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self, observation: Observation) -> np.ndarray:
        raise NotImplementedError
