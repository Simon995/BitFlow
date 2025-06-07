# BitFlow
A High-Performance Framework for PyTorch-Compatible Deep Learning Operators and Algorithms, Built with Python, CUDA, C++, and C

## 1. 设置开发环境
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh

## 2. 激活虚拟环境
source .venv/bin/activate
uv python install 3.11.5
uv python pin 3.11
uv sync

## 3. 构建项目
make build

## 4. 运行测试
make test

## 5. 创建分发包
python -m build

## 6. 安装本地包
pip install dist/bitflow-0.1.0-*.whl

## 7. 或者开发模式安装
pip install -e .
