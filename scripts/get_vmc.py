import click
import os.path as osp
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

current_dir = osp.dirname(osp.abspath(__file__))
pickle_dir = osp.join(current_dir, '../output/pkl')


def extract_index(filepath):
    matchs = re.findall(r'(\d+)', filepath)
    return int(matchs[-1])


def dict_to_df(d):
    dd = {(k1, k2): v2 for k1, v1 in d.items() for k2, v2 in v1.items()}
    df = pd.DataFrame.from_dict(dd, orient='index')
    df.index.names = ['theta_wind', 'theta_sail']
    for col in df.columns:
        if col == 'index':
            continue
        for i, new_col in enumerate(['min', 'max']):
            df[f'{col}_{new_col}'] = df[col].apply(lambda x: x[i])
        df = df.drop(col, axis=1)
    return df


def extract_vmc_from_df(df):
    df = df.copy()
    df.index = df.index.get_level_values('theta_wind')
    v_max = df['vmc_max'].groupby('theta_wind').max()
    v_max = np.maximum(0, v_max)
    return v_max.index, v_max.values


def get_vmc(wind_velocity):
    pathname = osp.join(pickle_dir, f'bounds_(v_wind_{wind_velocity}).*.pkl')
    filepaths = sorted(glob(pathname), key=extract_index, reverse=True)
    if not filepaths:
        print(
            f'Error: Please run `python3 extract_sim_bounds.py --wind-velocity={wind_velocity}` to extract the velocity bounds first.')
        return
    filepath = filepaths[0]
    d = pickle.load(open(filepath, 'rb'))
    df = dict_to_df(d)

    thetas, vmc = extract_vmc_from_df(df)
    thetas = np.deg2rad(thetas)
    vmc_dict = dict(zip(thetas, vmc))
    return vmc_dict


@click.command()
@click.option('--wind-velocity', default=1, help='Wind velocity', type=int)
def get_vmc_cmd(wind_velocity):
    vmc_dict = get_vmc(wind_velocity)
    output_path = osp.join(pickle_dir, f'vmc_(v_wind_{wind_velocity}).pkl')
    pickle.dump(vmc_dict, open(output_path, 'wb'))
    print(f'VMC for v_wind={wind_velocity} saved at {output_path}.')


if __name__ == '__main__':
    get_vmc_cmd()
