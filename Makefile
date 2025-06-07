.PHONY: help setup build test clean install benchmark docs format lint pre-commit

help:   ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:   ## 设置开发环境
	@chmod +x scripts/setup_dev.sh
	@./scripts/setup_dev.sh

build:   ## 构建和安装 BitFlow
	@echo "Cleaning previous build artifacts..."
	@rm -rf build/ dist/ *.egg-info/ bitflow.egg-info/ src/*.egg-info/ src/bitflow.egg-info/
	@echo "Building and installing BitFlow in editable mode using uv..."
	@uv python pin 3.11
	@uv pip install -e . -v
	@echo "Build and installation complete!"

test:   ## 运行测试
	@echo "Running BitFlow tests..."
	@uv python3 tests/test_ops/test_conv.py

clean:   ## 清理构建文件
	@echo "Cleaning build files..."
	@rm -rf build/ dist/ *.egg-info/
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find bitflow -type f -name "*.so" -delete

install:    ## 安装包
	@pip install -e .

dev-install:    ## 开发模式安装
	@pip install -e ".[dev]"

quick-build:    ## 快速构建
	@python setup.py build_ext --inplace