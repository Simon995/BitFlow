#!/bin/bash

set -e

echo "Building BitFlow..."

# 清理之前的构建
rm -rf build/ dist/ *.egg-info/

# 构建扩展
python setup.py build_ext --inplace

# 安装包
pip install -e . -v

echo "Build complete!"