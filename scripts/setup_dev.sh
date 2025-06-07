#!/bin/bash

set -e

echo "Setting up BitFlow development environment..."

# 1. 创建虚拟环境
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 升级pip
pip install --upgrade pip

# 4. 安装PyTorch (根据CUDA版本选择)
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 5. 安装构建工具
pip install ninja pybind11[global] cmake

# 6. 安装开发依赖
pip install -e ".[dev]"

# 7. 安装pre-commit hooks
pre-commit install

# 8. 初始化git submodules (如果有的话)
if [ -f ".gitmodules" ]; then
    git submodule update --init --recursive
fi

echo "Development environment setup complete!"
echo "Activate the environment with: source .venv/bin/activate"