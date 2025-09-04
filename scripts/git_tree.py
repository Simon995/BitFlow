#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def is_git_repo(path='.'):
    """检查当前目录是否是git仓库"""
    try:
        subprocess.run(['git', 'status'], 
                      cwd=path, 
                      capture_output=True, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_git_files(path='.'):
    """获取git跟踪的所有文件"""
    try:
        result = subprocess.run(['git', 'ls-files'], 
                               cwd=path, 
                               capture_output=True, 
                               text=True, 
                               check=True)
        return set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
    except subprocess.CalledProcessError:
        return set()

def should_exclude_dir(dir_path, exclude_dirs, git_files):
    """判断目录是否应该被排除"""
    dir_name = os.path.basename(dir_path)
    
    # 检查是否在排除列表中
    if dir_name in exclude_dirs:
        return True
    
    # 检查完整路径是否在排除列表中
    if dir_path in exclude_dirs:
        return True
    
    # 检查目录下是否有git跟踪的文件
    has_git_files = any(file_path.startswith(dir_path + '/') or file_path == dir_path 
                       for file_path in git_files)
    
    return not has_git_files

def print_tree(directory='.', exclude_dirs=None, prefix='', is_last=True, git_files=None):
    """
    递归打印目录树结构
    
    Args:
        directory: 要打印的目录路径
        exclude_dirs: 要排除的目录列表
        prefix: 当前行的前缀
        is_last: 是否是同级最后一个项目
        git_files: git跟踪的文件集合
    """
    if exclude_dirs is None:
        exclude_dirs = set()
    
    if git_files is None:
        git_files = get_git_files(directory)
    
    # 获取当前目录名
    dir_name = os.path.basename(os.path.abspath(directory))
    if not dir_name:
        dir_name = '.'
    
    # 打印当前目录
    connector = '└── ' if is_last else '├── '
    print(f"{prefix}{connector}{dir_name}/")
    
    # 计算新的前缀
    new_prefix = prefix + ('    ' if is_last else '│   ')
    
    try:
        # 获取目录内容
        items = []
        current_path = os.path.abspath(directory)
        
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            rel_path = os.path.relpath(item_path, '.')
            
            if os.path.isdir(item_path):
                # 检查目录是否应该被排除
                if not should_exclude_dir(rel_path, exclude_dirs, git_files):
                    items.append(('dir', item, item_path))
            else:
                # 检查文件是否被git跟踪
                if rel_path in git_files:
                    items.append(('file', item, item_path))
        
        # 排序：目录在前，文件在后，同类型按名称排序
        items.sort(key=lambda x: (x[0] == 'file', x[1].lower()))
        
        # 打印文件和目录
        for i, (item_type, item_name, item_path) in enumerate(items):
            is_last_item = (i == len(items) - 1)
            
            if item_type == 'dir':
                print_tree(item_path, exclude_dirs, new_prefix, is_last_item, git_files)
            else:
                connector = '└── ' if is_last_item else '├── '
                print(f"{new_prefix}{connector}{item_name}")
                
    except PermissionError:
        print(f"{new_prefix}[Permission Denied]")

def main():
    """主函数"""
    # 默认排除的目录
    default_exclude = {
        '.git',           # Git目录
        '__pycache__',    # Python缓存
        '.pytest_cache',  # Pytest缓存
        'node_modules',   # Node.js依赖
        '.vscode',        # VS Code配置
        '.idea',          # IntelliJ IDEA配置
        'venv',           # Python虚拟环境
        'env',            # Python虚拟环境
        '.env',           # 环境变量文件夹
        'dist',           # 构建目录
        'build',          # 构建目录
        '.DS_Store',      # macOS系统文件
        'Thumbs.db',      # Windows缩略图
    }
    
    # 可以在这里添加更多要排除的目录
    custom_exclude = {
        'logs',         # 日志目录
        'temp',         # 临时文件目录
        '.cache',       # 缓存目录
        '.venv',           # Python虚拟环境
        '.ruff_cache',    # Ruff缓存
        'bitflow.egg-info',  # Python包信息
        'third_party',   # 第三方库
    }
    
    # 合并排除列表
    exclude_dirs = default_exclude | custom_exclude
    
    # 检查是否是git仓库
    if not is_git_repo():
        print("错误: 当前目录不是Git仓库")
        return
    
    print("Git文件夹树结构:")
    print("=" * 50)
    
    # 打印树结构
    print_tree('.', exclude_dirs)
    
    print("\n排除的目录:")
    for exclude_dir in sorted(exclude_dirs):
        print(f"  - {exclude_dir}")

if __name__ == "__main__":
    main()