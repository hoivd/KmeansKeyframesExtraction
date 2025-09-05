# Base image có CUDA runtime + dev
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

WORKDIR /app

# Cài các dependency cơ bản
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv neovim && \
    rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch (GPU build) - dùng pip chính thức
RUN python3.10 -m venv venv
RUN /app/venv/bin/pip install --upgrade pip

# Cài PyTorch (GPU build)
RUN /app/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Cài đặt CuPy (CUDA 12.x)
RUN /app/venv/bin/pip install cupy-cuda12x -i https://mirrors.aliyun.com/pypi/simple

# Cài đặt RAPIDS cuML (cần NVIDIA PyPI index)
RUN /app/venv/bin/pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==23.6.*
RUN /app/venv/bin/pip install opencv-python

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0


# Kiểm tra phiên bản
# RUN python3 -c "import torch, cupy, cuml; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CuPy:', cupy.__version__); print('cuML:', cuml.__version__)"