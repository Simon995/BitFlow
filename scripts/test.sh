#!/bin/bash

set -e

echo "Running BitFlow tests..."

# 运行单元测试
python -m pytest tests/ -v --tb=short

# 运行性能测试
echo "Running benchmarks..."
python -m pytest tests/benchmarks/ -v --benchmark-only

# 运行集成测试
echo "Running integration tests..."
python examples/basic_usage.py

echo "All tests passed!"