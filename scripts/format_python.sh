#!/bin/bash
# scripts/format_python.sh - Python 代码格式化和检查脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Python 源码目录
PYTHON_DIRS=("bitflow" "tests" "examples" "scripts")

# 工具检查函数
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}错误: $1 未安装${NC}"
        echo "请安装 $1: $2"
        exit 1
    fi
}

# 检查 uv 并安装 ruff
check_and_install_ruff() {
    echo -e "${BLUE}检查 ruff 安装状态...${NC}"

    if ! uv run ruff --version &> /dev/null; then
        echo -e "${YELLOW}ruff 未安装，正在安装...${NC}"
        uv add ruff --dev
        echo -e "${GREEN}ruff 安装完成${NC}"
    else
        echo -e "${GREEN}ruff 已安装${NC}"
    fi
}

# 格式化 Python 代码
format_python() {
    echo -e "${BLUE}正在格式化 Python 代码...${NC}"

    for dir in "${PYTHON_DIRS[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            echo -e "${YELLOW}格式化 $dir${NC}"
            uv run ruff format "$PROJECT_ROOT/$dir"
        fi
    done

    # 格式化根目录的 Python 文件
    if find "$PROJECT_ROOT" -maxdepth 1 -name "*.py" -type f | grep -q .; then
        echo -e "${YELLOW}格式化根目录 Python 文件${NC}"
        uv run ruff format "$PROJECT_ROOT"/*.py
    fi

    echo -e "${GREEN}Python 代码格式化完成${NC}"
}

# 检查 Python 代码质量
lint_python() {
    echo -e "${BLUE}正在检查 Python 代码质量...${NC}"

    local has_errors=false
    local fix_mode="$1"

    for dir in "${PYTHON_DIRS[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            echo -e "${YELLOW}检查 $dir${NC}"

            if [ "$fix_mode" = "fix" ]; then
                if ! uv run ruff check --fix "$PROJECT_ROOT/$dir"; then
                    has_errors=true
                fi
            else
                if ! uv run ruff check "$PROJECT_ROOT/$dir"; then
                    has_errors=true
                fi
            fi
        fi
    done

    # 检查根目录的 Python 文件
    if find "$PROJECT_ROOT" -maxdepth 1 -name "*.py" -type f | grep -q .; then
        echo -e "${YELLOW}检查根目录 Python 文件${NC}"
        if [ "$fix_mode" = "fix" ]; then
            if ! uv run ruff check --fix "$PROJECT_ROOT"/*.py; then
                has_errors=true
            fi
        else
            if ! uv run ruff check "$PROJECT_ROOT"/*.py; then
                has_errors=true
            fi
        fi
    fi

    if [ "$has_errors" = true ]; then
        echo -e "${RED}发现代码质量问题${NC}"
        if [ "$fix_mode" != "fix" ]; then
            echo -e "${YELLOW}提示: 运行 '$0 lint-fix' 自动修复部分问题${NC}"
        fi
        exit 1
    else
        echo -e "${GREEN}Python 代码质量检查通过${NC}"
    fi
}

# 检查代码格式是否正确
check_format() {
    echo -e "${BLUE}检查 Python 代码格式...${NC}"

    local format_issues=false

    for dir in "${PYTHON_DIRS[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            echo -e "${YELLOW}检查 $dir 格式${NC}"
            if ! uv run ruff format --check "$PROJECT_ROOT/$dir"; then
                format_issues=true
            fi
        fi
    done

    # 检查根目录的 Python 文件
    if find "$PROJECT_ROOT" -maxdepth 1 -name "*.py" -type f | grep -q .; then
        echo -e "${YELLOW}检查根目录 Python 文件格式${NC}"
        if ! uv run ruff format --check "$PROJECT_ROOT"/*.py; then
            format_issues=true
        fi
    fi

    if [ "$format_issues" = true ]; then
        echo -e "${RED}发现格式问题，请运行: $0 format${NC}"
        exit 1
    else
        echo -e "${GREEN}代码格式检查通过${NC}"
    fi
}

# 运行类型检查
type_check() {
    echo -e "${BLUE}正在进行类型检查...${NC}"

    # 检查 mypy 是否可用
    if ! uv run mypy --version &> /dev/null; then
        echo -e "${YELLOW}mypy 未安装，跳过类型检查${NC}"
        return
    fi

    local has_errors=false

    for dir in "${PYTHON_DIRS[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ] && [ "$dir" != "tests" ]; then
            echo -e "${YELLOW}类型检查 $dir${NC}"
            if ! uv run mypy "$PROJECT_ROOT/$dir"; then
                has_errors=true
            fi
        fi
    done

    if [ "$has_errors" = true ]; then
        echo -e "${RED}发现类型检查问题${NC}"
        exit 1
    else
        echo -e "${GREEN}类型检查通过${NC}"
    fi
}

# 运行测试
run_tests() {
    echo -e "${BLUE}运行测试...${NC}"

    if [ -d "$PROJECT_ROOT/tests" ]; then
        uv run pytest
        echo -e "${GREEN}测试完成${NC}"
    else
        echo -e "${YELLOW}未找到测试目录，跳过测试${NC}"
    fi
}

# 生成代码覆盖率报告
coverage_report() {
    echo -e "${BLUE}生成代码覆盖率报告...${NC}"

    if [ -d "$PROJECT_ROOT/tests" ]; then
        uv run pytest --cov=bitflow --cov-report=html --cov-report=term
        echo -e "${GREEN}覆盖率报告生成完成 (htmlcov/index.html)${NC}"
    else
        echo -e "${YELLOW}未找到测试目录，跳过覆盖率报告${NC}"
    fi
}

# 显示项目状态
show_status() {
    echo -e "${PURPLE}=== 项目状态检查 ===${NC}"

    echo -e "${BLUE}Python 环境信息:${NC}"
    uv run python --version

    echo -e "\n${BLUE}已安装的包:${NC}"
    uv pip list | head -10

    echo -e "\n${BLUE}代码统计:${NC}"
    if command -v cloc &> /dev/null; then
        cloc --include-lang=Python bitflow/ tests/ 2>/dev/null || echo "使用 find 统计..."
        find bitflow/ tests/ -name "*.py" -type f | wc -l | xargs echo "Python 文件数:"
    else
        find bitflow/ tests/ -name "*.py" -type f | wc -l | xargs echo "Python 文件数:"
    fi
}

# 主函数
main() {
    cd "$PROJECT_ROOT"

    # 检查 uv 是否安装
    check_tool "uv" "参考: https://github.com/astral-sh/uv"

    case "${1:-help}" in
        "format")
            check_and_install_ruff
            format_python
            ;;
        "lint")
            check_and_install_ruff
            lint_python
            ;;
        "lint-fix")
            check_and_install_ruff
            lint_python "fix"
            ;;
        "check")
            check_and_install_ruff
            check_format
            ;;
        "type-check"|"mypy")
            type_check
            ;;
        "test")
            run_tests
            ;;
        "coverage")
            coverage_report
            ;;
        "status")
            show_status
            ;;
        "all")
            check_and_install_ruff
            echo -e "${PURPLE}=== 开始完整代码检查流程 ===${NC}"
            format_python
            lint_python "fix"
            type_check
            run_tests
            echo -e "${PURPLE}=== 完整检查流程完成 ===${NC}"
            ;;
        "ci")
            check_and_install_ruff
            echo -e "${PURPLE}=== CI 检查流程 ===${NC}"
            check_format
            lint_python
            type_check
            run_tests
            echo -e "${PURPLE}=== CI 检查完成 ===${NC}"
            ;;
        "help"|*)
            echo "用法: $0 {format|lint|lint-fix|check|type-check|test|coverage|status|all|ci}"
            echo ""
            echo "命令:"
            echo "  format      - 格式化 Python 代码 (ruff format)"
            echo "  lint        - 检查 Python 代码质量 (ruff check)"
            echo "  lint-fix    - 检查并自动修复 Python 代码问题"
            echo "  check       - 检查代码格式是否正确 (不修改文件)"
            echo "  type-check  - 运行类型检查 (mypy)"
            echo "  test        - 运行测试 (pytest)"
            echo "  coverage    - 生成代码覆盖率报告"
            echo "  status      - 显示项目状态信息"
            echo "  all         - 执行完整的开发流程 (格式化+检查+测试)"
            echo "  ci          - 执行 CI 检查流程 (仅检查，不修改)"
            echo "  help        - 显示此帮助"
            echo ""
            echo "示例:"
            echo "  $0 format           # 格式化所有 Python 代码"
            echo "  $0 lint-fix         # 检查并自动修复问题"
            echo "  $0 all              # 完整的开发前检查"
            echo "  $0 ci               # CI 环境检查"
            ;;
    esac
}

main "$@"