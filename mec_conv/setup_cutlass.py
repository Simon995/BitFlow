from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os # 导入 os 模块以帮助处理路径

# --- 重要：设置此路径 ---
# 设置您的 CUTLASS 库安装的根目录路径。
# 例如，如果 CUTLASS 克隆在 /home/user/cutlass，此路径将是 /home/user/cutlass
# 请确保此路径对您的系统是正确的。
CUTLASS_ROOT_DIR = "/mnt/d/code/CodeHub/cplusplus/Memory-efficient-Convolution-for-Deep-Neural-Network-master/cutlass" # <--- 请修改此行为您的实际 CUTLASS 根路径

# 从根目录构造包含路径
CUTLASS_INCLUDE_DIR = os.path.join(CUTLASS_ROOT_DIR, "include")
CUTLASS_TOOLS_UTIL_INCLUDE_DIR = os.path.join(CUTLASS_ROOT_DIR, "tools", "util", "include")

# 您也可以使用环境变量来设置此根路径，例如：
# CUTLASS_ROOT_DIR_ENV = os.environ.get("CUTLASS_ROOT_DIR")
# if CUTLASS_ROOT_DIR_ENV:
#    CUTLASS_ROOT_DIR = CUTLASS_ROOT_DIR_ENV # 如果环境变量已设置，则使用它
#    CUTLASS_INCLUDE_DIR = os.path.join(CUTLASS_ROOT_DIR, "include")
#    CUTLASS_TOOLS_UTIL_INCLUDE_DIR = os.path.join(CUTLASS_ROOT_DIR, "tools", "util", "include")
# else:
#    # 如果环境变量未设置，则使用上面定义的默认路径
#    # 确保上面的 CUTLASS_ROOT_DIR 仍然是您需要修改的默认值
#    if CUTLASS_ROOT_DIR == "/path/to/your/cutlass_root": # 检查是否仍为占位符
#        print("Error: CUTLASS_ROOT_DIR is not set and the default placeholder is being used.")
#        print("Please set the CUTLASS_ROOT_DIR variable in setup_cutlass.py or the CUTLASS_ROOT_DIR environment variable.")
#        exit(1)


# 检查路径是否有效
if not os.path.exists(CUTLASS_ROOT_DIR):
    print(f"Error: CUTLASS_ROOT_DIR ('{CUTLASS_ROOT_DIR}') does not exist.")
    print("Please set it correctly in setup_cutlass.py or as an environment variable.")
    exit(1)
if not os.path.exists(CUTLASS_INCLUDE_DIR) or not os.path.exists(os.path.join(CUTLASS_INCLUDE_DIR, "cutlass")):
    print(f"Warning: CUTLASS_INCLUDE_DIR ('{CUTLASS_INCLUDE_DIR}') does not seem to be a valid CUTLASS include directory.")
    print("Please ensure CUTLASS_ROOT_DIR is set correctly.")
    # exit(1) # 根据需要决定是否在此处退出

if not os.path.exists(CUTLASS_TOOLS_UTIL_INCLUDE_DIR) or not os.path.exists(os.path.join(CUTLASS_TOOLS_UTIL_INCLUDE_DIR, "cutlass", "util")):
    print(f"Warning: CUTLASS_TOOLS_UTIL_INCLUDE_DIR ('{CUTLASS_TOOLS_UTIL_INCLUDE_DIR}') does not seem to be valid.")
    print("This path is often needed for CUTLASS utilities like host_tensor.h.")
    # exit(1) # 根据需要决定是否在此处退出


setup(
    name='mec_conv_cuda',  # Python 模块的名称，import 时使用
    ext_modules=[
        CUDAExtension(
            name='mec_conv_cuda',  # 必须与上面的 name 匹配
            sources=[
                'mec_conv_wrapper.cpp',  # C++ 包装器源文件
                'mec_conv.cu',         # CUDA 源文件 (应包含使用 CUTLASS 的卷积实现)
            ],
            include_dirs=[
                CUTLASS_INCLUDE_DIR,
                CUTLASS_TOOLS_UTIL_INCLUDE_DIR,
                # 如果需要，在此处添加其他包含目录
            ],
            extra_compile_args={
                'cxx': [
                    '-g',       # 生成调试信息
                    '-O2',      # 优化级别
                    '-std=c++17' # CUTLASS 通常需要 C++17
                ],
                'nvcc': [
                    '-O2',      # 优化级别
                    '-std=c++17', # 确保 nvcc 对主机和设备代码也使用 C++17 以保持一致性
                    # 为 Ada Lovelace 架构 (例如 RTX 4070) 设置
                    # CUDA 11.8 及更高版本支持 sm_89
                    '-gencode=arch=compute_89,code=sm_89',
                    # '--verbose', # 取消注释以获取编译期间更详细的 nvcc 输出
                    # '-Xcompiler', '-Wall', # 通过 nvcc 将 -Wall 传递给主机编译器的示例
                    # '-Xcudafe', '--diag_suppress=esa_on_defaulted_function_ignored', # 示例：抑制特定警告
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension  # 指定使用 PyTorch 的 BuildExtension
    }
)

# 关于 CUTLASS_ROOT_DIR 的说明:
# 如果您的 CUTLASS 头文件结构如下:
# /path/to/cutlass_root/
#   ├── include/                <-- CUTLASS_INCLUDE_DIR 指向这里
#   │   └── cutlass/
#   │       └── gemm/
#   │           └── device/
#   │               └── gemm_universal.h
#   └── tools/
#       └── util/
#           └── include/        <-- CUTLASS_TOOLS_UTIL_INCLUDE_DIR 指向这里
#               └── cutlass/util/host_tensor.h
# 那么 CUTLASS_ROOT_DIR 应该设置为 "/path/to/cutlass_root"。
# 脚本将自动构造 CUTLASS_INCLUDE_DIR 和 CUTLASS_TOOLS_UTIL_INCLUDE_DIR。
# `.cu` 文件中的 `#include "cutlass/gemm/device/gemm_universal.h"` 将能正确解析。
