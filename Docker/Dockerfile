# 使用多阶段构建
# 第一阶段：编译和安装Python 3.7.13
FROM ubuntu:18.04 as builder

# 更新apt源并安装必要的工具
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    mpich

# 复制本地的Python-3.7.13.tgz文件到镜像中
COPY Python-3.7.13.tgz /tmp/

# 解压并安装Python 3.7.13
RUN cd /tmp && tar -xzf Python-3.7.13.tgz && cd Python-3.7.13 && ./configure --enable-optimizations --prefix=/opt/python && make altinstall

# 第二阶段：创建最终的镜像
FROM python:3.7.13-slim-buster

# 安装必要的工具
RUN apt-get update && apt-get install -y \
    git \
    sudo \
    wget

# 复制第一阶段安装好的Python到最终的镜像中
COPY --from=builder /opt/python /opt/python

# 设置Python 3.7.13为默认的python版本
RUN update-alternatives --install /usr/bin/python python /opt/python/bin/python3.7 1

# 安装pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

# 设置bash为默认的指令
CMD ["/bin/bash"]