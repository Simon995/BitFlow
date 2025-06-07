from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mec_conv3d_cuda',  # Python module name, used with import
    ext_modules=[
        CUDAExtension(
            name='mec_conv3d_cuda',  # Must match the name above
            sources=[
                'mec_conv3d_wrapper.cpp',  # C++ wrapper source file for 3D convolution
                'mec_conv3d.cu',         # CUDA source file for 3D convolution
            ],
            # Additional compile arguments can be added for CXX (C++ compiler) and NVCC (CUDA compiler)
            extra_compile_args={'cxx': ['-g', '-O3'], # Example: debug info and optimization level 2
                                'nvcc': ['-O3', # Example: optimization level 2
                                         # Set for Ada Lovelace architecture (e.g., RTX 4070)
                                         # CUDA 11.8 and later support sm_89
                                         # CUDA 12.x natively supports newer architectures
                                         '-gencode=arch=compute_89,code=sm_89',
                                         # You can optionally retain compatibility for older architectures,
                                         # but this will increase compile time.
                                         # Example: '-gencode=arch=compute_75,code=sm_75', # Turing
                                         ]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension  # Specify using PyTorch's BuildExtension
    }
)

# Notes:
# 1. Place 'mec_conv3d_wrapper.cpp' (containing the C++ binding code from the artifact "pytorch_cpp_extension_3d")
#    and 'mec_conv3d.cu' (containing the CUDA kernel code from the artifact "cuda_conv3d")
#    in the same directory as this setup.py file.
# 2. The '-gencode' argument in extra_compile_args is crucial as it specifies the target GPU's compute capability.
#    - For RTX 4070 (Ada Lovelace architecture), the corresponding compute capability is sm_89.
#    - We have included '-gencode=arch=compute_89,code=sm_89'.
# 3. Ensure your PyTorch version was compiled with CUDA 12.x (e.g., CUDA 12.1 or a version compatible with your system's CUDA).
#    If PyTorch was compiled with an older CUDA version (like 11.x), you might encounter CUDA version mismatch issues
#    when compiling the extension, even if your system CUDA is newer.
#    You can check the CUDA version used by PyTorch with torch.version.cuda.
#
#    Common architecture codes:
#    - Pascal (GTX 10xx): sm_60, sm_61
#    - Volta (V100): sm_70
#    - Turing (RTX 20xx, T4): sm_75
#    - Ampere (RTX 30xx, A100): sm_80, sm_86
#    - Ada Lovelace (RTX 40xx): sm_89
#    - Hopper (H100): sm_90
