from setuptools import setup, find_packages

setup(
    name='sailboat-gym',
    version='1.0.1',
    author='Lucas Marandat',
    author_email='lucas.mrdt+sailboat@gmail.com',
    description='Dynamic simulation environment for sailboats. With Sailboat Gym, you can explore and experiment with different control algorithms and strategies in a realistic virtual sailing environment.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lucasmrdt/sailboat-gym',
    packages=find_packages(),
    install_requires=[
        'gymnasium==0.28.1',
        'msgpack_python==0.5.6',
        'numpy==1.24.3',
        'pydantic==1.10.7',
        'pyzmq==25.0.2',
        'tqdm==4.65.0',
        'opencv-python==4.7.0.72',
        'imageio-ffmpeg==0.4.8',
        'docker==6.1.2',
        'moviepy==1.0.3'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
