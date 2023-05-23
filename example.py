import numpy as np
import tqdm
from itertools import count
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from sail_gym import CV2DRenderer


def ctrl(_):
    rudder_angle = np.deg2rad(-5)
    sail_angle = np.deg2rad(-45)
    return {'theta_rudder': np.array(rudder_angle), 'theta_sail': np.array(sail_angle)}


def generate_wind(seed):
    np.random.seed(seed)
    return np.random.normal(0, 2, 2)


env = gym.make('SailboatLSAEnv-v0',
               renderer=CV2DRenderer(),
               wind_generator_fn=generate_wind,
               container_tag='mss6',
               keep_sim_alive=True)
env = RecordVideo(env, video_folder='./output/videos/')

truncated = False
obs, info = env.reset(seed=10)
env.render()
for _ in tqdm.tqdm(count(), desc='Running simulation'):
    obs, reward, terminated, truncated, info = env.step(ctrl(obs))
    if truncated:
        break
    env.render()
env.close()
