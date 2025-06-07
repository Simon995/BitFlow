cmake_minimum_required(VERSION 3.18)
project(bitflow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 查找必要的包
find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# 设置 CUDA 架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89")
endif()

# 包含目录
include_directories(csrc/include)

# 收集源文件
file(GLOB_RECURSE CPP_SOURCES "csrc/src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "csrc/src/*.cu")

# 创建扩展模块
pybind11_add_module(_C ${CPP_SOURCES} ${CUDA_SOURCES} csrc/python/bind.cpp)

# 链接库
target_link_libraries(_C PRIVATE "${TORCH_LIBRARIES}")

# 编译选项
target_compile_definitions(_C PRIVATE VERSION_INFO="${EXAMPLE_VERSION_INFO}")

# CUDA 编译选项
if(CUDA_SOURCES)
  set_property(TARGET _C PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# 设置输出目录
set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                    "${CMAKE_CURRENT_SOURCE_DIR}/bitflow/_C")
