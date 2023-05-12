import numpy as np
import atexit
from typing import Callable, TypedDict

from ...interfaces import IRenderer
from ...types import Observation, Action
from ...utils import is_debugging
from ..env import SailboatEnv
from .lsa_sim import LSASim


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class Vector2(TypedDict):
    x: float
    y: float


class SimObservation(TypedDict):
    p_boat: Vector3
    dt_p_boat: Vector3
    theta_boat: Vector3
    dt_theta_boat: Vector3
    theta_rudder: float
    dt_theta_rudder: float
    theta_sail: float
    dt_theta_sail: float
    wind: Vector2


class SailboatLSAEnv(SailboatEnv):
    SIM_RATE = 10  # Hz

    def __init__(self, reward_fn: Callable[[Observation, Action], float] = lambda *_: 0, renderer: IRenderer = None, wind_generator_fn: Callable[[int], np.ndarray] = None, video_speed: float = 1, keep_sim_alive: bool = False, docker_tag: str = 'default'):
        """Sailboat LSA environment

        Args:
            reward_fn (Callable[[Observation, Action], float], optional): Use a custom reward function depending of your task. Defaults to lambda *_: 0.
            renderer (IRenderer, optional): Renderer instance to be used for rendering the environment, look at sailboat_gym/renderers folder for more information. Defaults to None.
            wind_generator_fn (Callable[[int], np.ndarray], optional): Function that returns a 2D vector representing the global wind during the simulation. Defaults to None.
            video_speed (float, optional): Speed of the video recording. Defaults to 1.
            keep_sim_alive (bool, optional): Keep the simulation running even after the program exits. Defaults to False.
            docker_tag (str, optional): Docker tag to be used for the simulation, see the documentation for more information. Defaults to 'default'.
        """
        super().__init__()

        # IMPORTANT: The following variables are required by the gymnasium API
        self.render_mode = renderer.get_render_mode() if renderer else None
        self.metadata = {
            'render_modes': renderer.get_render_modes() if renderer else [],
            'render_fps': video_speed * self.SIM_RATE,
        }

        self.sim = LSASim(docker_tag)
        self.reward_fn = reward_fn
        self.renderer = renderer
        self.obs = None
        self.wind_generator_fn = wind_generator_fn

        # Stop the simulation when the program exits
        if not keep_sim_alive:
            atexit.register(self.sim.stop)

    def __parse_sim_obs(self, obs: SimObservation) -> Observation:
        return {
            'p_boat': np.array([obs['p_boat']['x'], obs['p_boat']['y'], obs['p_boat']['z']], dtype=np.float32),
            'dt_p_boat': np.array([obs['dt_p_boat']['x'], obs['dt_p_boat']['y'], obs['dt_p_boat']['z']], dtype=np.float32),
            'theta_boat': np.array([obs['theta_boat']['x'], obs['theta_boat']['y'], obs['theta_boat']['z']], dtype=np.float32),
            'dt_theta_boat': np.array([obs['dt_theta_boat']['x'], obs['dt_theta_boat']['y'], obs['dt_theta_boat']['z']], dtype=np.float32),
            'theta_rudder': np.array([obs['theta_rudder']], dtype=np.float32),
            'dt_theta_rudder': np.array([obs['dt_theta_rudder']], dtype=np.float32),
            'theta_sail': np.array([obs['theta_sail']], dtype=np.float32),
            'dt_theta_sail': np.array([obs['dt_theta_sail']], dtype=np.float32),
            'wind': np.array([obs['wind']['x'], obs['wind']['y']], dtype=np.float32),
        }

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        np.random.seed(seed)

        # generate wind
        if self.wind_generator_fn:
            wind = self.wind_generator_fn(seed)
        else:
            wind = np.random.normal(0, 2, 2)

        obs, info = self.sim.reset(wind, self.SIM_RATE)
        self.obs = self.__parse_sim_obs(obs)

        # setup the renderer, its needed to know the min/max position of the boat
        if self.renderer:
            assert 'min_position' in info and 'max_position' in info, 'Missing min/max position in info'
            min_position = np.array(
                [info['min_position']['x'], info['min_position']['y'], info['min_position']['z']], dtype=np.float32)
            max_position = np.array(
                [info['max_position']['x'], info['max_position']['y'], info['max_position']['z']], dtype=np.float32)
            self.renderer.setup(min_position, max_position)

        if is_debugging():
            print('\nResetting environment:')
            print(f'  -> Wind: {wind}')
            print(f'  -> frequency: {self.SIM_RATE} Hz')
            print(f'  <- Obs: {self.obs}')
            print(f'  <- Info: {info}')

        return self.obs, info

    def step(self, action: Action):
        obs, terminated, info = self.sim.step(action)
        self.obs = self.__parse_sim_obs(obs)
        reward = self.reward_fn(self.obs, action)

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
