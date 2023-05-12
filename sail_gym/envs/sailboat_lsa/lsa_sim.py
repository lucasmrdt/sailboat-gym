import docker
import zmq
import msgpack
import time
import numpy as np

from ...utils import ProfilingMeta, is_debugging, DurationProgress


class LSASim(metaclass=ProfilingMeta):
    PORT = 5555  # set in Dockerfile

    def __init__(self, docker_tag='default') -> None:
        if is_debugging():
            print('[LSASim] Launching docker container')
        self.container = self.__launch_or_get_docker(docker_tag)
        self.__wait_until_ready()
        self.socket = self.__create_connection()

    def reset(self, wind: np.ndarray[2], sim_rate: int):
        self.__send_msg({
            'reset': {
                'wind': {'x': wind[0], 'y': wind[1]},
                'freq': sim_rate,
            }
        })
        msg = self.__recv_msg()
        return msg['obs'], msg['info']

    def step(self, action):
        self.__send_msg({'action': action})
        msg = self.__recv_msg()
        return msg['obs'], msg['done'], msg['info']

    def close(self):
        self.__send_msg({'close': True})
        self.__recv_msg()

    def stop(self):
        with DurationProgress(total=12, desc='Stopping docker container'):
            self.container.stop()

    def __launch_or_get_docker(self, docker_tag):
        with DurationProgress(total=7, desc='Launching docker container'):
            client = docker.from_env()
            name = f'sailboat-sim-lsa-gym-{docker_tag}'

            # try to find an existing container with the given name
            if client.containers.list(filters={'name': name}):
                container = client.containers.get(name)
            else:
                container = None
            if container:
                if container.status == 'running':
                    if is_debugging():
                        print('[LSASim] Found existing docker container')
                    return container
                else:
                    container.remove()

            # kill all other containers which are bound to the same port
            for container in client.containers.list():
                ports = container.ports
                if ports and f'{self.PORT}/tcp' in ports:
                    container.kill()

            # launch a new container if none found or the existing container is not running
            container = client.containers.run(
                f'lucasmrdt/sailboat-sim-lsa-gym:{docker_tag}',
                detach=True,
                name=name,
                ports={
                    f'{self.PORT}/tcp': self.PORT
                },
                remove=True,
            )

            if is_debugging():
                print('[LSASim] Launched new docker container')

        return container

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
        socket.connect(f'tcp://localhost:{self.PORT}')
        return socket

    def __send_msg(self, msg):
        self.socket.send(msgpack.packb(msg))

    def __recv_msg(self):
        msg = msgpack.unpackb(self.socket.recv(), raw=False)
        if 'error' in msg:
            raise Exception(msg['error'])
        return msg
