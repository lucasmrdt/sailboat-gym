import numpy as np
import tqdm
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo

from sailboat_gym import CV2DRenderer, Observation, NB_STEPS_PER_SECONDS

EPISODE_LENGTH = NB_STEPS_PER_SECONDS * 60 * 1  # 3 minutes


def ctrl(obs: Observation):
    wanted_heading = np.deg2rad(-30)
    rudder_angle = obs['theta_boat'][2] - wanted_heading
    sail_angle = np.deg2rad(-90)
    return {'theta_rudder': np.array(rudder_angle), 'theta_sail': np.array(sail_angle)}


def generate_wind(_):
    theta_wind = np.deg2rad(-(90+180))
    wind_speed = 2
    return np.array([np.cos(theta_wind), np.sin(theta_wind)]) * wind_speed


env = gym.make('SailboatLSAEnv-v0',
               renderer=CV2DRenderer(),
               wind_generator_fn=generate_wind,
               container_tag='mss1',
               video_speed=10,
               keep_sim_alive=True)
env = TimeLimit(env, max_episode_steps=EPISODE_LENGTH)
env = RecordVideo(env, video_folder='./output/videos/')

truncated = False
obs, info = env.reset(seed=10)
env.render()
for t in tqdm.trange(EPISODE_LENGTH, desc='Running simulation'):
    obs, reward, terminated, truncated, info = env.step(ctrl(obs))
    if terminated or truncated:
        break
    env.render()
env.close()
