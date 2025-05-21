`uv` 是一个现代化的 Python 包管理器，旨在提供快速、可靠的依赖管理和虚拟环境操作。以下是 `uv` 的常用指令及其功能介绍，简洁明了：

### 1. **包管理相关指令**
- **`uv add <package>`**  
  安装指定包到当前项目的虚拟环境中，并更新 `pyproject.toml` 的依赖列表。  
  示例：`uv add requests`  
  - 可指定版本：`uv add "requests>=2.28.0"`

- **`uv remove <package>`**  
  从项目中卸载指定包，并更新 `pyproject.toml`。  
  示例：`uv remove requests`

- **`uv sync`**  
  同步项目的虚拟环境，确保环境中安装的包与 `pyproject.toml` 中的依赖一致。  
  示例：`uv sync`

- **`uv lock`**  
  生成或更新 `uv.lock` 文件，锁定项目依赖的精确版本。  
  示例：`uv lock`

- **`uv install`**  
  安装 `pyproject.toml` 或 `requirements.txt` 中列出的依赖到虚拟环境。  
  示例：`uv install`

### 2. **虚拟环境管理**
- **`uv venv`**  
  创建一个新的虚拟环境（默认在 `.venv` 目录）。  
  示例：`uv venv`  
  - 指定名称：`uv venv myenv`

- **`uv activate`**  
  激活虚拟环境（通常手动激活，如 `source .venv/bin/activate`）。  
  注：`uv` 本身不提供 `activate` 命令，依赖 shell 命令。

### 3. **运行和脚本执行**
- **`uv run <command>`**  
  在项目的虚拟环境中运行命令或脚本，自动确保依赖已安装。  
  示例：`uv run python script.py`  
  - 常用于运行 `manage.py` 或其他脚本。

- **`uv python <command>`**  
  使用指定 Python 版本运行命令。  
  示例：`uv python --version`

### 4. **Python 版本管理**
- **`uv python list`**  
  列出所有可用的 Python 版本。  
  示例：`uv python list`

- **`uv python install <version>`**  
  安装指定的 Python 版本。  
  示例：`uv python install 3.11`

- **`uv python pin <version>`**  
  为项目指定使用的 Python 版本。  
  示例：`uv python pin 3.11`

### 5. **其他实用指令**
- **`uv init`**  
  初始化一个新项目，创建 `pyproject.toml` 文件。  
  示例：`uv init myproject`

- **`uv tree`**  
  显示项目依赖树。  
  示例：`uv tree`

- **`uv export`**  
  导出依赖到 `requirements.txt` 文件。  
  示例：`uv export > requirements.txt`

- **`uv cache clean`**  
  清理 `uv` 的缓存目录，释放磁盘空间。  
  示例：`uv cache clean`

### 6. **全局选项**
- **`--python <version>`**  
  指定使用的 Python 版本。  
  示例：`uv add requests --python 3.10`

- **`--no-sync`**  
  跳过同步虚拟环境。  
  示例：`uv add requests --no-sync`

- **`-q` / `--quiet`**  
  减少命令输出，静默执行。  
  示例：`uv sync -q`

### 注意事项
- **项目配置文件**：`uv` 主要依赖 `pyproject.toml` 管理依赖，推荐使用符合 PEP 621 标准的格式。
- **虚拟环境**：默认在项目目录下的 `.venv` 中创建，激活方式与传统 Python 一致。
- **性能**：`uv` 比 `pip` 快得多，适合大型项目。
- **兼容性**：支持 `requirements.txt`，但更推荐使用 `pyproject.toml`。

如果需要更详细的解释或特定场景的用法，请告诉我！