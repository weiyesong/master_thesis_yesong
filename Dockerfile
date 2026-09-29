# Development image for the EO UQ thesis experiments.
#
# Matches the first-stage EuroSAT RGB baseline:
# - Python 3.11 from the official PyTorch image
# - PyTorch 2.5.1 + CUDA 12.4
# - torchvision for EuroSAT RGB and pretrained ResNet18
# - TorchGeo / Lightning-UQ-Box kept available for later DOFA and UQ work
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

# System packages used by remote-sensing and vision libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    vim \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Keep files created in the bind-mounted workspace owned by the host user.
ARG USERNAME=yesong
ARG USER_UID=1025
ARG USER_GID=1025
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && mkdir -p /workspace /tmp/matplotlib \
    && chown -R ${USERNAME}:${USERNAME} /workspace /tmp/matplotlib

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
    numpy \
    pandas \
    pyarrow \
    scikit-learn \
    matplotlib \
    seaborn \
    pyyaml \
    tqdm \
    jupyterlab \
    tensorboard \
    pillow \
    opencv-python-headless \
    torchgeo \
    'GeoBenchV2 @ git+https://github.com/The-AI-Alliance/GEO-Bench-2.git@fd9d0b664e6fb0faba54636bdff4906634debd4b' \
    torchmetrics \
    lightning \
    lightning-uq-box \
    timm \
    kornia \
    rasterio \
    wandb

USER ${USERNAME}

CMD ["/bin/bash"]
