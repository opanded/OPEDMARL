# 使用多阶段构建
# 第一阶段：创建Python 3.7.13环境
FROM python:3.7.13-slim-buster as builder

# 更新apt源并安装必要的工具
RUN apt-get update && apt-get install -y \
    git \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 克隆OPEDMARL项目
RUN git clone https://github.com/opanded/OPEDMARL /OPEDMARL

# 第二阶段：创建最终的镜像
FROM python:3.7.13-slim-buster

RUN apt-get update && apt-get install -y \
    git \
    sudo \
    wget

# 复制第一阶段创建好的Python环境到最终的镜像中
COPY --from=builder /usr/local /usr/local

# 安装pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && git clone https://github.com/opanded/OPEDMARL /OPEDMARL

# 设置bash为默认的指令
CMD ["/bin/bash"]

# 设置默认工作目录
WORKDIR /OPEDMARL
