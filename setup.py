from setuptools import setup, find_packages
from setuptools.command.install import install
import os

class CustomInstall(install):
    def run(self):
        # 安装apt依赖项
        os.system("apt install mpich")
        # 安装pip依赖项
        os.system("pip install tensorflow==1.13.1 numpy==1.16.4 gym==0.13.0 mpi4py==3.1.4 protobuf==3.19.4 imageio==2.21.1 matplotlib==3.5.3 joblib==1.1.0")
        # 调用父类的run方法
        install.run(self)

setup(name='iros22_darl1n',
      version='0.0.1',
      description='Distributed multi-agent Reinforcement Learning with One-Hope Neighbors (DARL1N)',
      author='BaoqianWang',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      cmdclass={'install': CustomInstall} # 指定自定义安装类
)
