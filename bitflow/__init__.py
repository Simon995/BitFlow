"""
BitFlow - 高性能深度学习框架
"""

__version__ = "0.1.0"

# 导入 C++ 扩展
try:
    from . import _C

    print("BitFlow C++ extension loaded successfully")
except ImportError:
    print("BitFlow C++ extension not found")
    _C = None

# 导入主要模块
from .ops import conv2d

__all__ = ["conv2d", "_C", "__version__"]
