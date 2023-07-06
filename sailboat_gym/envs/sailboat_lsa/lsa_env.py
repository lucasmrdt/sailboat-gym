import numpy as np
import atexit
from typing import Callable

from ...abstracts import AbcRender
from ...types import Observation, Action
from ...utils import is_debugging
from ..env import SailboatEnv
from .lsa_sim import LSASim


class SailboatLSAEnv(SailboatEnv):
    NB_STEPS_PER_SECONDS = 10  # Hz

    def __init__(self, reward_fn: Callable[[Observation, Action, Observation], float] = lambda *_: 0, renderer: AbcRender = None, wind_generator_fn: Callable[[int], np.ndarray] = None, video_speed: float = 1, keep_sim_alive: bool = False, container_tag: str = 'mss1', name='default'):
        """Sailboat LSA environment

        Args:
            reward_fn (Callable[[Observation, Action], float], optional): Use a custom reward function depending of your task. Defaults to lambda *_: 0.
            renderer (AbcRender, optional): Renderer instance to be used for rendering the environment, look at sailboat_gym/renderers folder for more information. Defaults to None.
            wind_generator_fn (Callable[[int], np.ndarray], optional): Function that returns a 2D vector representing the global wind during the simulation. Defaults to None.
            video_speed (float, optional): Speed of the video recording. Defaults to 1.
            keep_sim_alive (bool, optional): Keep the simulation running even after the program exits. Defaults to False.
            container_tag (str, optional): Docker tag to be used for the simulation, see the documentation for more information. Defaults to 'mss1'.
            name ([type], optional): Name of the simulation, required to run multiples environment on same machine.. Defaults to 'default'.
        """
        super().__init__()

        # IMPORTANT: The following variables are required by the gymnasium API
        self.render_mode = renderer.get_render_mode() if renderer else None
        self.metadata = {
            'render_modes': renderer.get_render_modes() if renderer else [],
            'render_fps': video_speed * self.NB_STEPS_PER_SECONDS,
        }

        self.sim = LSASim(container_tag, name)
        self.reward_fn = reward_fn
        self.renderer = renderer
        self.obs = None
        self.wind_generator_fn = wind_generator_fn

        # Stop the simulation when the program exits
        if not keep_sim_alive:
            atexit.register(self.sim.stop)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        np.random.seed(seed)

        # generate wind
        if self.wind_generator_fn:
            wind = self.wind_generator_fn(seed)
        else:
            wind = np.random.normal(0, 2, 2)

        self.obs, info = self.sim.reset(wind, self.NB_STEPS_PER_SECONDS)

        # setup the renderer, its needed to know the min/max position of the boat
        if self.renderer:
            self.renderer.setup(info['map_bounds'])

        if is_debugging():
            print('\nResetting environment:')
            print(f'  -> Wind: {wind}')
            print(f'  -> frequency: {self.NB_STEPS_PER_SECONDS} Hz')
            print(f'  <- Obs: {self.obs}')
            print(f'  <- Info: {info}')

        return self.obs, info

    def step(self, action: Action):
        next_obs, terminated, info = self.sim.step(action)
        reward = self.reward_fn(self.obs, action, next_obs)
        self.obs = next_obs

        if is_debugging():
            print('\nStepping environment:')
            print(f'  -> Action: {action}')
            print(f'  <- Obs: {self.obs}')
            print(f'  <- Reward: {reward}')
            print(f'  <- Terminated: {terminated}')
            print(f'  <- Info: {info}')

        return self.obs, reward, terminated, False, info

    def render(self):
        assert self.renderer, 'No renderer'
        return self.renderer.render(self.obs)

    def close(self):
        self.sim.close()
        self.obs = None
