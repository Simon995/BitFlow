python代码检查主要功能

代码格式化 (format) - 使用 ruff format 格式化 Python 代码
代码质量检查 (lint) - 使用 ruff check 检查代码质量
自动修复 (lint-fix) - 自动修复可修复的代码问题
格式检查 (check) - 检查格式是否正确，不修改文件
类型检查 (type-check) - 使用 mypy 进行类型检查
运行测试 (test) - 运行 pytest 测试
覆盖率报告 (coverage) - 生成代码覆盖率报告
项目状态 (status) - 显示项目信息
完整流程 (all) - 执行所有开发检查
CI 流程 (ci) - 仅检查，适合 CI 环境

使用方法
bash# 给脚本添加执行权限
chmod +x scripts/format_python.sh

# 使用示例
./scripts/format_python.sh format      # 格式化代码
./scripts/format_python.sh lint-fix    # 检查并修复问题
./scripts/format_python.sh all         # 完整开发流程
./scripts/format_python.sh ci          # CI 检查流程
特色功能

自动安装 ruff - 如果未安装会自动使用 uv 安装
彩色输出 - 使用颜色区分不同类型的信息
错误处理 - 遇到问题会适当退出并提示
灵活配置 - 可以轻松修改要检查的目录
CI 友好 - 提供专门的 CI 检查模式

脚本会自动检查 bitflow/、tests/、examples/、scripts/ 目录以及根目录的 Python 文件。你可以根据项目结构修改 PYTHON_DIRS 数组。