import numpy as np
import tqdm
from itertools import count
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from sailboat_gym import CV2DRenderer


def ctrl(_):
    rudder_angle = np.deg2rad(-20)
    sail_angle = np.deg2rad(+20)
    return {'theta_rudder': np.array(rudder_angle), 'theta_sail': np.array(sail_angle)}


def generate_wind(seed):
    return [-2, 0]


# mss4 -> x8-x9
# mss5 -> x9-x10
# mss6 -> x8-x10
# mss8 -> x6-x10
env = gym.make('SailboatLSAEnv-v0',
               renderer=CV2DRenderer(),
               wind_generator_fn=generate_wind,
               container_tag='mss6',
               video_speed=10)
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
