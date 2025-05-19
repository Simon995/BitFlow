from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mec_conv_cuda',  # Python 模块的名称，import 时使用
    ext_modules=[
        CUDAExtension(
            name='mec_conv_cuda',  # 必须与上面的 name 匹配
            sources=[
                'mec_conv_wrapper.cpp',  # C++ 包装器源文件
                'mec_conv.cu',         # CUDA 源文件
            ],
            # 可以为 CXX (C++编译器) 和 NVCC (CUDA编译器) 添加额外的编译参数
            extra_compile_args={'cxx': ['-g', '-O2'], # 示例：调试信息和优化级别2
                                'nvcc': ['-O2', # 示例：优化级别2
                                         # 为 Ada Lovelace 架构 (例如 RTX 4070) 设置
                                         # CUDA 11.8 及更高版本支持 sm_89
                                         # CUDA 12.x 天然支持更新的架构
                                         '-gencode=arch=compute_89,code=sm_89',
                                         # 您也可以选择性地为旧架构保留兼容性，但这会增加编译时间
                                         # 例如: '-gencode=arch=compute_75,code=sm_75', # Turing
                                         ]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension  # 指定使用 PyTorch 的 BuildExtension
    }
)

# 注意：
# 1. 将 'mec_conv_wrapper.cpp' 和 'mec_conv.cu' 文件放在与 setup.py 相同的目录下。
# 2. extra_compile_args 中的 '-gencode' 参数非常重要，它指定了目标 GPU 的计算能力。
#    - 对于 RTX 4070 (Ada Lovelace 架构)，对应的计算能力是 sm_89。
#    - 我们已经将 '-gencode=arch=compute_89,code=sm_89' 添加进去。
# 3. 请确保您的 PyTorch 版本是使用 CUDA 12.x (例如 CUDA 12.1 或与您系统 CUDA 12.8 兼容的版本) 编译的。
#    如果 PyTorch 是用较旧的 CUDA 版本（如 11.x）编译的，即使系统 CUDA 是 12.8，编译扩展时仍可能遇到 CUDA 版本不匹配的问题。
#    您可以通过 torch.version.cuda 查看 PyTorch 使用的 CUDA 版本。
#
#    常见的架构代号：
#    - Pascal (GTX 10xx): sm_60, sm_61
#    - Volta (V100): sm_70
#    - Turing (RTX 20xx, T4): sm_75
#    - Ampere (RTX 30xx, A100): sm_80, sm_86
#    - Ada Lovelace (RTX 40xx): sm_89
#    - Hopper (H100): sm_90
