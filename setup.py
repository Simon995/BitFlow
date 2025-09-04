# setup.py — 多算子可扩展 & 兼容性加强版
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from pathlib import Path
import os
import platform
import torch

# 项目根目录（绝对路径）
ROOT = Path(__file__).parent.resolve()

# 环境开关
FORCE_CPU = os.environ.get("BITFLOW_FORCE_CPU", "0") == "1"

# 与 PyTorch 对齐的 C++ ABI 宏


def torch_uses_cxx11_abi() -> int:
    try:
        from torch._C import _GLIBCXX_USE_CXX11_ABI  # type: ignore
        return int(_GLIBCXX_USE_CXX11_ABI)
    except Exception:
        return 0  # 保守缺省（PyTorch manylinux 轮子通常为 0）


CXX11_ABI = torch_uses_cxx11_abi()

# 递归收集源码：返回“相对 ROOT 的 POSIX 路径”，以避免 setuptools 的绝对路径报错


def rglob_sources():
    src_root = ROOT / "csrc"
    if not src_root.exists():
        raise RuntimeError(f"Missing source dir: {src_root}")

    exts_cpp = {".c", ".cc", ".cpp", ".cxx"}
    exts_cu = {".cu"}  # .cuh 为头文件，不编译

    cpp_sources, cu_sources = [], []
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in exts_cpp:
            cpp_sources.append(p)
        elif suf in exts_cu:
            cu_sources.append(p)

    def to_rel(p): return p.relative_to(ROOT).as_posix()
    return [to_rel(p) for p in cpp_sources], [to_rel(p) for p in cu_sources]


CPP_SOURCES, CU_SOURCES = rglob_sources()

print(f"Found C/C++ sources: {len(CPP_SOURCES)}")
for s in CPP_SOURCES:
    print(f"  - {s}")
if CU_SOURCES:
    print(f"Found CUDA sources: {len(CU_SOURCES)}")
    for s in CU_SOURCES:
        print(f"  - {s}")

# CUDA 架构选择


def default_cuda_archs():
    env = os.environ.get("BITFLOW_CUDA_ARCHS")
    if env:
        return [a.strip() for a in env.replace(",", ";").split(";") if a.strip()]
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return [f"{major}{minor}"]
    except Exception:
        pass
    return ["75", "80", "86", "89"]


ARCH_LIST = default_cuda_archs()


def nvcc_arch_flags(archs):
    flags = []
    for a in archs:
        flags += [f"-gencode=arch=compute_{a},code=sm_{a}"]
    return flags


# 编译/链接参数
IS_WINDOWS = platform.system() == "Windows"

CXX_ARGS_COMMON = [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]
if IS_WINDOWS:
    CXX_ARGS = ["/O2", "/std:c++17", "/openmp"]
    NVCC_ARGS = [
        "-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr",
        "-DWITH_CUDA",
        *nvcc_arch_flags(ARCH_LIST),
        f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}",
    ]
    BASE_LIBRARIES = []
    EXTRA_LINK_ARGS = []
else:
    CXX_ARGS = ["-O3", "-std=c++17", "-fopenmp", *CXX_ARGS_COMMON]
    NVCC_ARGS = [
        "-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr",
        "-lineinfo", "-DWITH_CUDA",
        *nvcc_arch_flags(ARCH_LIST),
        f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}",
    ]
    BASE_LIBRARIES = ["gomp"]
    TORCH_LIB = (Path(torch.__file__).parent / "lib").as_posix()
    EXTRA_LINK_ARGS = [f"-Wl,-rpath,{TORCH_LIB}"]

# include 目录改为“绝对路径”，以适配 Ninja 的临时构建目录


def collect_include_dirs_abs():
    incs = set()
    for d in [
        ROOT,                                       # 需要时支持 #include "bitflow/..."
        ROOT / "csrc",
        ROOT / "csrc" / "include",
        ROOT / "csrc" / "include" / "bitflow" / "ops",  # 你当前头文件包含层级
        ROOT / "third_party" / "cutlass" / "include",
    ]:
        if d.exists():
            incs.add(d.resolve().as_posix())

    csrc_dir = ROOT / "csrc"
    if csrc_dir.exists():
        for sub in csrc_dir.rglob("*"):
            if sub.is_dir():
                try:
                    if any(
                        p.is_file() and p.suffix.lower() in {
                            ".h", ".hpp", ".cuh"}
                        for p in sub.iterdir()
                    ):
                        incs.add(sub.resolve().as_posix())
                except PermissionError:
                    pass
    return sorted(incs)


INCLUDE_DIRS_ABS = collect_include_dirs_abs()

# 选择扩展类型（CPU/CUDA）
CUDA_OK = (not FORCE_CPU) and torch.cuda.is_available() and bool(CU_SOURCES)
if CUDA_OK:
    ExtensionClass = CUDAExtension
    SOURCES = CPP_SOURCES + CU_SOURCES     # 相对路径
    print("Building with CUDA support")
    EXTRA_COMPILE = {"cxx": CXX_ARGS + ["-DWITH_CUDA"], "nvcc": NVCC_ARGS}
    LIBRARIES = BASE_LIBRARIES + ["cublas", "curand", "cusparse"]
else:
    ExtensionClass = CppExtension
    SOURCES = CPP_SOURCES                  # 相对路径
    if FORCE_CPU:
        print("BITFLOW_FORCE_CPU=1 -> CPU-only build")
    elif not torch.cuda.is_available():
        print("CUDA not available -> CPU-only build")
    elif not CU_SOURCES:
        print("No CUDA sources found -> CPU-only build")
    EXTRA_COMPILE = {"cxx": CXX_ARGS}
    LIBRARIES = BASE_LIBRARIES

if not SOURCES:
    raise RuntimeError(
        "No source files found under csrc/. Add operators first.")

ext_modules = [
    ExtensionClass(
        name="bitflow._C",
        sources=SOURCES,                    # 相对路径（避免 setuptools 绝对路径报错）
        include_dirs=INCLUDE_DIRS_ABS,      # 绝对路径（保证在临时 build 目录也能找到头文件）
        extra_compile_args=EXTRA_COMPILE,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=[
            ("TORCH_EXTENSION_NAME", "_C"),
            ("TORCH_API_INCLUDE_EXTENSION_H", None),
            ("BITFLOW_WITH_CUDA", int(CUDA_OK)),
            ("_GLIBCXX_USE_CXX11_ABI", CXX11_ABI),
        ],
        libraries=LIBRARIES,
        language="c++",
    )
]

# 可选：包内打包共享库；若你已在 pyproject.toml 配置 package-data，可删掉这里
PKG_DATA = {
    "bitflow": [
        "bitflow/_C/*.so",
        "bitflow/_C/*.pyd",
        "bitflow/_C/*.dll",
        "bitflow/_C/*.dylib",
    ]
}

setup(
    name="bitflow",
    version="0.1.0",
    description="BitFlow: High-performance deep learning operators (PyTorch-compatible)",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",   # 如仅支持特定版本可收紧
        "numpy",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    package_data=PKG_DATA,  # 与 pyproject.toml 二选一
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
