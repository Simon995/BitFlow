# BitFlow

A High-Performance Framework for PyTorch-Compatible Deep Learning Operators and Algorithms, Built with Python, CUDA, C++, and C

## 1. 设置和激活虚拟环境(首先安装 uv 环境管理)

curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11.5
uv python pin 3.11
source .venv/bin/activate
uv sync

## 2. 构建项目

make build

## 3. 运行测试

make test

## 4. 构建 pip 包

uv run python -m build --no-isolation --wheel

## 5. 安装本地包

uv pip install dist/\*.whl

## 6. 或者开发模式安装

pip install -e .
