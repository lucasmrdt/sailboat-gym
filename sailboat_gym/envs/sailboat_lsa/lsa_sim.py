import docker
import zmq
import msgpack
import time
import datetime
import threading
import numpy as np
from typing import TypedDict

from ...utils import ProfilingMeta, is_debugging, DurationProgress
from ...types import Action, Observation, ResetInfo


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


class SimResetInfo(TypedDict):
    min_position: Vector3
    max_position: Vector3


class LSASim(metaclass=ProfilingMeta):
    DEFAULT_PORT = 5555  # set in Dockerfile

    def __init__(self, container_tag='mss1', name='default') -> None:
        self.container_tag = container_tag
        self.name = name

        self.wind = None
        self.sim_rate = None
        self.last_step_time = None
        self.container = None
        self.port = None
        self.socket = None
        self.is_running = False
        self.is_close = False

        self.__init_simulation()
        threading.Thread(target=self.__keep_trying_to_pause_simulation_if_not_used, daemon=True).start()  # noqa

    def reset(self, wind: np.ndarray[2], sim_rate: int):
        self.last_step_time = datetime.datetime.now()
        self.resume_if_needed()

        self.__send_msg({
            'reset': {
                'wind': {'x': wind[0], 'y': wind[1]},
                'freq': sim_rate,
            }
        })
        msg = self.__recv_msg()
        obs = self.__parse_sim_obs(msg['obs'])
        info = self.__parse_sim_reset_info(msg['info'])
        return obs, info

    def step(self, action: Action):
        self.last_step_time = datetime.datetime.now()
        self.resume_if_needed()

        self.__send_msg({
            'action': {
                'theta_rudder': action['theta_rudder'].item(),
                'theta_sail': action['theta_sail'].item(),
            }
        })
        msg = self.__recv_msg()
        obs = self.__parse_sim_obs(msg['obs'])
        done = msg['done']
        return obs, done, msg['info']

    def close(self):
        self.resume_if_needed()
        self.__send_msg({'close': True})
        self.__recv_msg()

    def stop(self):
        with DurationProgress(total=5, desc='Stopping docker container'):
            self.container.kill()

    def pause_if_needed(self):
        if self.is_running:
            self.container.pause()
            self.is_running = False

    def resume_if_needed(self):
        if not self.is_running:
            self.container.unpause()
            self.is_running = True

    def __init_simulation(self):
        if is_debugging():
            print('[LSASim] Launching docker container')
        self.container, self.port, self.is_running = self.__launch_or_get_container(self.container_tag, self.name)  # noqa
        self.__wait_until_ready()
        self.socket = self.__create_connection()
        self.pause_if_needed()

    def __keep_trying_to_pause_simulation_if_not_used(self):
        while not self.is_close:
            if is_debugging():
                print('[LSASim] Checking if simulation is used')
            if self.last_step_time and (datetime.datetime.now() - self.last_step_time).total_seconds() > 5:
                if is_debugging():
                    print('[LSASim] Simulation is not used, pausing it')
                self.pause_if_needed()
                self.last_step_time = None
            time.sleep(2)

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

    def __parse_sim_reset_info(self, info: SimResetInfo) -> ResetInfo:
        min_pos = np.array([info['min_position']['x'],
                            info['min_position']['y'],
                            0])
        max_pos = np.array([info['max_position']['x'],
                            info['max_position']['y'],
                            1])
        map_bounds = np.array([min_pos, max_pos], dtype=np.float32)
        return {
            'map_bounds': map_bounds,
        }

    def __get_available_port(self):
        def get_random_port():
            # https://stackoverflow.com/a/46023565
            return np.random.randint(49152, 65535)

        port = get_random_port()
        while True:
            try:
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.bind(f'tcp://*:{port}')
                socket.close()
                return port
            except zmq.error.ZMQError:
                port = get_random_port()

    def __launch_or_get_container(self, container_tag, name):
        with DurationProgress(total=7, desc='Launching docker container'):
            try:
                client = docker.from_env()
            except docker.errors.DockerException as e:
                if is_debugging():
                    raise e
                raise RuntimeError(
                    'Docker socket is not detected. Please start docker and try again or make sure that you have correctly installed docker (MacOS: refer to this instruction https://stackoverflow.com/a/76125150).') from e

            name = f'sailboat-sim-lsa-gym-{container_tag}-{name}'

            # try to find an existing container with the given name
            try:
                container = client.containers.get(name)
            except docker.errors.NotFound:
                container = None
            if container:
                is_running = container.status == 'running'
                port = container.attrs['NetworkSettings']['Ports'][
                    f'{self.DEFAULT_PORT}/tcp'][0]['HostPort']
                if is_debugging():
                    print(
                        f'\n[LSASim] Found existing docker container {name} running on port {port}')
                return container, port, is_running

            # find an available port
            port = self.__get_available_port()

            # launch a new container if none found or the existing container is not running
            try:
                container = client.containers.run(
                    f'lucasmrdt/sailboat-sim-lsa-gym:{container_tag}',
                    detach=True,
                    name=name,
                    ports={
                        f'{self.DEFAULT_PORT}/tcp': port,
                        '22/tcp': None,
                    },
                    remove=True,
                )
            except docker.errors.NotFound as e:
                raise RuntimeError(
                    f'Could not find docker image lucasmrdt/sailboat-sim-lsa-gym:{container_tag}. '
                    'Please make sure the image exists and try again.'
                ) from e
            except docker.errors.APIError as e:
                raise RuntimeError(
                    f'Error communicating with Docker API: {str(e)}. '
                    'Please check that the Docker daemon is running and try again.'
                ) from e
            except docker.errors.ContainerError as e:
                raise RuntimeError(
                    f'Container exited with non-zero exit code: {str(e)}. '
                    'Please check the container logs for more information.'
                ) from e
            except docker.errors.ImageNotFound as e:
                raise RuntimeError(
                    f'Image not found: {str(e)}. '
                    'Please check that the image exists on the Docker registry and try again.'
                ) from e

            if is_debugging():
                print('\n[LSASim] Launched new docker container')

        is_running = True
        return container, port, is_running

    def __wait_until_ready(self):
        with DurationProgress(total=17, desc='Waiting for docker container to be ready'):
            while True:
                logs = self.container.logs().decode('utf-8')
                if 'INTENTIFIED CONTROL!' in logs:
                    break
                time.sleep(1)

    def __create_connection(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f'tcp://localhost:{self.port}')
        return socket

    def __send_msg(self, msg):
        self.socket.send(msgpack.packb(msg))

    def __recv_msg(self):
        msg = msgpack.unpackb(self.socket.recv(), raw=False)
        if 'error' in msg:
            raise RuntimeError(msg['error'])
        return msg
