#!/bin/bash
# scripts/format_cpp.sh - C++/CUDA 代码格式化和检查脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 源码目录
CPP_DIRS=("csrc/src" "csrc/include" "csrc/cuda" "csrc/pybind")

# 工具检查函数
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}错误: $1 未安装${NC}"
        echo "请安装 $1: $2"
        exit 1
    fi
}

# 格式化 C++/CUDA 代码
format_cpp() {
    echo -e "${BLUE}正在格式化 C++/CUDA 代码...${NC}"

    for dir in "${CPP_DIRS[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            echo -e "${YELLOW}格式化 $dir${NC}"
            find "$PROJECT_ROOT/$dir" -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" -o -name "*.cuh" \) \
                -exec clang-format -i {} \;
        fi
    done

    echo -e "${GREEN}C++/CUDA 代码格式化完成${NC}"
}

# 检查 C++/CUDA 代码
lint_cpp() {
    echo -e "${BLUE}正在检查 C++/CUDA 代码...${NC}"

    local has_errors=false

    for dir in "${CPP_DIRS[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            echo -e "${YELLOW}检查 $dir${NC}"

            # 使用 clang-tidy 检查
            find "$PROJECT_ROOT/$dir" -type f \( -name "*.cpp" -o -name "*.cu" \) | while read -r file; do
                if ! clang-tidy "$file" -- -I"$PROJECT_ROOT/csrc/include" -std=c++17; then
                    has_errors=true
                fi
            done
        fi
    done

    if [ "$has_errors" = true ]; then
        echo -e "${RED}发现代码质量问题${NC}"
        exit 1
    else
        echo -e "${GREEN}C++/CUDA 代码检查通过${NC}"
    fi
}

# 格式化 CMake 文件
# format_cmake() {
#     echo -e "${BLUE}正在格式化 CMake 文件...${NC}"

#     find "$PROJECT_ROOT" -name "CMakeLists.txt" -o -name "*.cmake" | while read -r file; do
#         echo -e "${YELLOW}格式化 $file${NC}"
#         cmake-format -i "$file"
#     done

#     echo -e "${GREEN}CMake 文件格式化完成${NC}"
# }

# 检查格式是否正确
check_format() {
    echo -e "${BLUE}检查代码格式...${NC}"

    local format_issues=false

    for dir in "${CPP_DIRS[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            find "$PROJECT_ROOT/$dir" -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" -o -name "*.cuh" \) | while read -r file; do
                if ! clang-format --dry-run --Werror "$file" &>/dev/null; then
                    echo -e "${RED}格式问题: $file${NC}"
                    format_issues=true
                fi
            done
        fi
    done

    if [ "$format_issues" = true ]; then
        echo -e "${RED}发现格式问题，请运行: $0 format${NC}"
        exit 1
    else
        echo -e "${GREEN}代码格式检查通过${NC}"
    fi
}

# 主函数
main() {
    cd "$PROJECT_ROOT"

    case "${1:-help}" in
        "format")
            check_tool "clang-format" "sudo apt install clang-format 或 brew install clang-format"
            check_tool "cmake-format" "pip install cmakelang"
            format_cpp
            # format_cmake
            ;;
        "lint")
            check_tool "clang-tidy" "sudo apt install clang-tidy 或 brew install llvm"
            lint_cpp
            ;;
        "check")
            check_tool "clang-format" "sudo apt install clang-format 或 brew install clang-format"
            check_format
            ;;
        "all")
            check_tool "clang-format" "sudo apt install clang-format 或 brew install clang-format"
            check_tool "clang-tidy" "sudo apt install clang-tidy 或 brew install llvm"
            check_tool "cmake-format" "pip install cmakelang"
            format_cpp
            # format_cmake
            lint_cpp
            ;;
        "help"|*)
            echo "用法: $0 {format|lint|check|all}"
            echo ""
            echo "命令:"
            echo "  format  - 格式化 C++/CUDA/CMake 代码"
            echo "  lint    - 检查 C++/CUDA 代码质量"
            echo "  check   - 检查代码格式是否正确"
            echo "  all     - 执行所有操作"
            echo "  help    - 显示此帮助"
            ;;
    esac
}

main "$@"