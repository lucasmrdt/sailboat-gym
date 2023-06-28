import numpy as np
import tqdm
from itertools import count
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo

from sailboat_gym import CV2DRenderer, EPISODE_LENGTH, Observation


def ctrl(obs: Observation):
    wanted_heading = 0
    rudder_angle = obs['theta_boat'][2] - wanted_heading
    sail_angle = np.deg2rad(-30)
    return {'theta_rudder': np.array(rudder_angle), 'theta_sail': np.array(sail_angle)}


def generate_wind(_):
    theta_wind = np.deg2rad(-(90+180))
    wind_speed = 3
    return np.array([np.cos(theta_wind), np.sin(theta_wind)]) * wind_speed


env = gym.make('SailboatLSAEnv-v0',
               renderer=CV2DRenderer(),
               wind_generator_fn=generate_wind,
               container_tag='mss1',
               video_speed=20,
               keep_sim_alive=True)
env = TimeLimit(env, max_episode_steps=EPISODE_LENGTH)
env = RecordVideo(env, video_folder='./output/videos/')

truncated = False
obs, info = env.reset(seed=10)
env.render()
for _ in tqdm.tqdm(count(), desc='Running simulation'):
    obs, reward, terminated, truncated, info = env.step(ctrl(obs))
    if terminated or truncated:
        break
    env.render()
env.close()
