import sys
sys.path.append('..')  # noqa
sys.path.append('.')  # noqa

import tqdm
import pickle
import click
import os.path as osp
import numpy as np
import gymnasium as gym
from itertools import count
from collections import defaultdict
from gymnasium.wrappers.time_limit import TimeLimit

from sailboat_gym import CV2DRenderer, EPISODE_LENGTH

current_dir = osp.dirname(osp.abspath(__file__))


global_theta_wind = 0
global_wind_velocity = None


def generate_wind(_):
    theta_wind_rad = np.deg2rad(global_theta_wind)
    theta_wind_rad += np.pi  # wind is pointing to the boat
    theta_wind_rad *= -1  # we simulate boat rotation of theta_wind_rad
    return np.array([np.cos(theta_wind_rad), np.sin(theta_wind_rad)])*global_wind_velocity


env = gym.make('SailboatLSAEnv-v0',
               renderer=CV2DRenderer(),
               wind_generator_fn=generate_wind,
               container_tag='mss5',
               keep_sim_alive=True)
env = TimeLimit(env, max_episode_steps=EPISODE_LENGTH/2)


def get_vmc(obs):
    v = obs['dt_p_boat'][0:2]
    d = np.array([1, 0])  # x-axis
    return np.dot(v, d)


def deep_convert_to_dict(d):
    d = dict(d)
    for k, v in d.items():
        if isinstance(v, defaultdict):
            d[k] = deep_convert_to_dict(v)
    return d


def save_bounds(bounds, idx):
    bounds = deep_convert_to_dict(bounds)
    file_path = osp.join(
        current_dir,
        f'../output/pkl/bounds_(v_wind_{global_wind_velocity}).{idx}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(bounds, f)
    print(f'Saved bounds to file: {file_path}')


def run_simulation(bounds, theta_wind, theta_sail):
    global global_theta_wind
    global_theta_wind = theta_wind

    def ctrl(_):
        sail_angle = np.deg2rad(theta_sail)
        return {'theta_rudder': np.array(0), 'theta_sail': np.array(sail_angle)}

    obs, info = env.reset(seed=0)
    for _ in tqdm.tqdm(count(), desc='Running simulation'):
        obs, reward, terminated, truncated, info = env.step(ctrl(obs))

        vmc = get_vmc(obs)
        v_min, v_max = bounds[theta_wind][theta_sail]['vmc']
        bounds[theta_wind][theta_sail]['vmc'] = (
            min(v_min, vmc), max(v_max, vmc))

        for k, v in obs.items():
            if k in ['p_boat']:
                for d in range(v.shape[0]):
                    v_min, v_max = bounds[theta_wind][theta_sail][f'{k}_{d}']
                    bounds[theta_wind][theta_sail][f'{k}_{d}'] = (
                        min(v_min, v[d]), max(v_max, v[d]))
            else:
                v = np.linalg.norm(v)
                v_min, v_max = bounds[theta_wind][theta_sail][k]
                bounds[theta_wind][theta_sail][k] = (
                    min(v_min, v), max(v_max, v))

        if terminated or truncated:
            break
    env.close()


@click.command()
@click.option('--wind-velocity', default=1, help='Wind velocity', type=int)
def extract_sim_stats(wind_velocity):
    global global_wind_velocity
    global_wind_velocity = int(wind_velocity)

    bounds_by_wind_by_sail_by_var = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: (np.inf, -np.inf))))

    for i, theta_wind in enumerate(range(0, 360, 5)):
        print('\n'+'-'*80)
        print(f'wind angle: {theta_wind}')
        for theta_sail in range(-90, 90+1, 5):
            print(f'\tsail angle: {theta_sail}')
            for _ in range(4):
                run_simulation(bounds_by_wind_by_sail_by_var,
                               theta_wind,
                               theta_sail)
                break
            break
        save_bounds(bounds_by_wind_by_sail_by_var, i)
        break


if __name__ == '__main__':
    extract_sim_stats()
