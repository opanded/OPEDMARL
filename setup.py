from setuptools import setup, find_packages

setup(name='iros22_darl1n',
      version='0.0.1',
      description='Distributed multi-agent Reinforcement Learning with One-Hope Neighbors (DARL1N)',
      author='BaoqianWang',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      license='MIT', # 添加了license参数
      install_requires=['tensorflow==1.13.1', 'numpy==1.16.4', 'gym==0.13.0', 'mpi4py==3.1.4', 'protobuf==3.19.4', 'imageio==2.21.1', 'matplotlib==3.5.3', 'joblib==1.1.0'], # 添加了install_requires参数
      python_requires='>=3.7,<3.8' # 添加了python_requires参数
)
