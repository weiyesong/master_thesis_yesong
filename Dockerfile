# 使用官方带 CUDA 支持的 PyTorch 基础镜像 (Python 3.11, PyTorch 2.5)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 设置工作目录
WORKDIR /workspace

# 设置环境变量，避免安装时出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 1. 安装系统级依赖
# 遥感和计算机视觉通常需要以下库：
# - libgl1-mesa-glx: OpenCV 等图像处理库通常需要它
# - gdal-bin, libgdal-dev: 处理地理空间数据 (GeoTIFF等) 的核心库
# - libspatialindex-dev: 某些遥感库 (如 Rtree) 的依赖
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    libgl1-mesa-glx \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 库
# 重点安装: torchgeo, lightning-uq-box
# 以及常用的 tensorboard (用于查看训练曲线)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab \
    tensorboard \
    opencv-python-headless \
    pillow \
    torchgeo \
    lightning-uq-box \
    pytorch-lightning \
    torchmetrics \
    wandb \
    timm \
    kornia

# 默认进入 bash
CMD ["/bin/bash"]