from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os
import glob
import platform

def check_cuda_available():
    """检查CUDA是否可用"""
    try:
        return torch.cuda.is_available() and torch.version.cuda is not None
    except Exception:
        return False

def find_source_files(root_dir):
    """递归查找所有C/C++和CUDA源文件"""
    csrc_dir = os.path.join(root_dir, "csrc")
    
    if not os.path.exists(csrc_dir):
        print(f"Warning: {csrc_dir} does not exist")
        return [], []
    
    # 查找所有C/C++源文件
    cpp_patterns = [
        os.path.join(csrc_dir, "**", "*.c"),     # C源文件
        os.path.join(csrc_dir, "**", "*.cpp"),   # C++源文件
        os.path.join(csrc_dir, "**", "*.cc"),    # C++源文件
        os.path.join(csrc_dir, "**", "*.cxx"),   # C++源文件
    ]
    
    cpp_sources = []
    for pattern in cpp_patterns:
        cpp_sources.extend(glob.glob(pattern, recursive=True))
    
    # 查找所有CUDA源文件
    cuda_patterns = [
        os.path.join(csrc_dir, "**", "*.cu"),
        os.path.join(csrc_dir, "**", "*.cuh"),
    ]
    
    cuda_sources = []
    for pattern in cuda_patterns:
        found_files = glob.glob(pattern, recursive=True)
        # 只包含.cu文件，.cuh文件是头文件不需要编译
        cuda_sources.extend([f for f in found_files if f.endswith('.cu')])
    
    return cpp_sources, cuda_sources

def get_compile_args():
    """根据平台获取编译参数"""
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        # Windows编译参数
        cxx_args = ["/O2", "/std:c++17", "/openmp"]
        nvcc_args = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-DWITH_CUDA",
            # 添加多个架构支持
            "-gencode=arch=compute_75,code=sm_75",  # Turing
            "-gencode=arch=compute_80,code=sm_80",  # Ampere
            "-gencode=arch=compute_86,code=sm_86",  # Ampere (RTX 30xx)
            "-gencode=arch=compute_89,code=sm_89",  # RTX 40xx
        ]
        libraries = []
    else:
        # Linux/macOS编译参数
        cxx_args = ["-O3", "-std=c++17", "-fopenmp"]
        nvcc_args = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-DWITH_CUDA",
            # 添加多个架构支持
            "-gencode=arch=compute_75,code=sm_75",  # Turing
            "-gencode=arch=compute_80,code=sm_80",  # Ampere
            "-gencode=arch=compute_86,code=sm_86",  # Ampere (RTX 30xx)
            "-gencode=arch=compute_89,code=sm_89",  # RTX 40xx
        ]
        libraries = ["gomp"]
    
    return cxx_args, nvcc_args, libraries

# 获取项目根目录的绝对路径
root_dir = os.path.dirname(os.path.abspath(__file__))

# 自动查找源文件
cpp_sources, cuda_sources = find_source_files(root_dir)

print(f"Found C/C++ sources: {len(cpp_sources)}")
for src in cpp_sources:
    print(f"  - {os.path.relpath(src, root_dir)}")

if cuda_sources:
    print(f"Found CUDA sources: {len(cuda_sources)}")
    for src in cuda_sources:
        print(f"  - {os.path.relpath(src, root_dir)}")

# 包含目录 - 确保所有必要的路径都包含
include_dirs = [
    root_dir,  # 项目根目录，用于 bitflow/ 头文件
]

# 添加csrc相关目录
csrc_include = os.path.join(root_dir, "csrc", "include")
csrc_root = os.path.join(root_dir, "csrc")
if os.path.exists(csrc_include):
    include_dirs.append(csrc_include)
if os.path.exists(csrc_root):
    include_dirs.append(csrc_root)

# 添加第三方库目录
cutlass_include = os.path.join(root_dir, "third_party", "cutlass", "include")
if os.path.exists(cutlass_include):
    include_dirs.append(cutlass_include)

# 添加所有子目录到包含路径
csrc_dir = os.path.join(root_dir, "csrc")
if os.path.exists(csrc_dir):
    for root, dirs, files in os.walk(csrc_dir):
        # 检查是否包含头文件
        if any(f.endswith(('.h', '.hpp', '.cuh')) for f in files):
            include_dirs.append(root)

# 获取编译参数
cxx_args, nvcc_args, base_libraries = get_compile_args()

# 选择扩展类型
cuda_available = check_cuda_available()
has_cuda_sources = len(cuda_sources) > 0

if cuda_available and has_cuda_sources:
    extension = CUDAExtension
    sources = cpp_sources + cuda_sources
    print("Building with CUDA support")
    extra_compile_args = {
        "cxx": cxx_args + ["-DWITH_CUDA"],
        "nvcc": nvcc_args,
    }
    # CUDA库
    cuda_libraries = ["cublas", "curand", "cusparse"]
    libraries = base_libraries + cuda_libraries
else:
    extension = CppExtension
    sources = cpp_sources
    if not cuda_available:
        print("CUDA not available, building CPU-only version")
    elif not has_cuda_sources:
        print("No CUDA sources found, building CPU-only version")
    else:
        print("Building CPU-only version")
    extra_compile_args = {"cxx": cxx_args}
    libraries = base_libraries

# 检查是否有源文件
if not sources:
    raise RuntimeError("No source files found! Please check your csrc directory structure.")

# 设置链接参数
extra_link_args = []
if platform.system() != "Windows":
    # 获取 PyTorch 库路径
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    extra_link_args = [f"-Wl,-rpath,{torch_lib_path}"]

ext_modules = [
    extension(
        name="bitflow._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ("TORCH_EXTENSION_NAME", "_C"),
            ("TORCH_API_INCLUDE_EXTENSION_H", None),
        ],
        libraries=libraries,
        language="c++",
    )
]

setup(
    name="bitflow",
    version="0.1.0",
    description="BitFlow: High-performance quantization library",
    author="Your Name",
    author_email="your.email@example.com",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "numpy",
    ],
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)