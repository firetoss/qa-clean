# Conda 环境管理指南

本文档介绍如何使用 conda 管理 QA Clean 项目的环境。

## 📋 前置要求

### 1. 安装 Anaconda 或 Miniconda

**Anaconda (推荐):**
- 下载地址: https://www.anaconda.com/products/distribution
- 包含大量预编译的科学计算包
- 适合数据科学和机器学习

**Miniconda:**
- 下载地址: https://docs.conda.io/en/latest/miniconda.html
- 轻量级版本，只包含 conda 和 Python
- 适合有经验的用户

### 2. 验证安装

```bash
conda --version
```

## 🚀 快速开始

### 创建基础环境

```bash
# 克隆项目
git clone <repository-url>
cd qa-clean

# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate qa-clean

# 安装项目
pip install -e .
```

### 创建开发环境

```bash
# 创建开发环境
conda env create -f environment-dev.yml

# 激活环境
conda activate qa-clean-dev

# 安装项目
pip install -e ".[dev]"
```

## 🔧 环境管理命令

### 查看环境

```bash
# 列出所有环境
conda env list

# 查看当前环境
conda info --envs
```

### 激活/停用环境

```bash
# 激活环境
conda activate qa-clean

# 停用环境
conda deactivate
```

### 更新环境

```bash
# 更新所有包
conda update --all

# 更新特定包
conda update pandas numpy
```

### 删除环境

```bash
# 删除环境
conda env remove -n qa-clean

# 删除环境及其所有包
conda remove -n qa-clean --all
```

## 📦 包管理

### 安装包

```bash
# 使用 conda 安装
conda install package_name

# 使用 pip 安装（在激活的环境中）
pip install package_name

# 从特定 channel 安装
conda install -c conda-forge package_name
```

### 查看包

```bash
# 列出当前环境的所有包
conda list

# 查看特定包
conda list package_name
```

### 导出环境

```bash
# 导出环境配置
conda env export > environment_backup.yml

# 导出精确版本
conda env export --from-history > environment_exact.yml
```

## 🐍 Python 版本管理

### 查看 Python 版本

```bash
python --version
conda list python
```

### 切换 Python 版本

```bash
# 安装特定版本的 Python
conda install python=3.9

# 创建新环境时指定版本
conda create -n qa-clean python=3.9
```

## 🔍 故障排除

### 常见问题

1. **环境激活失败**
   ```bash
   # 重新初始化 conda
   conda init
   # 重启终端
   ```

2. **包安装冲突**
   ```bash
   # 清理缓存
   conda clean --all
   # 重新创建环境
   conda env remove -n qa-clean
   conda env create -f environment.yml
   ```

3. **GPU 相关包安装失败**
   ```bash
   # 确保使用正确的 channel
   conda install -c pytorch pytorch
   pip install faiss-gpu
   ```

### 环境检查

```bash
# 检查环境状态
conda info

# 检查包依赖
conda list --export

# 检查环境变量
conda env config vars list
```

## 📚 最佳实践

### 1. 环境命名
- 使用描述性名称
- 避免使用空格和特殊字符
- 包含项目名称和用途

### 2. 依赖管理
- 优先使用 conda 安装包
- 对于 conda 没有的包，使用 pip
- 定期更新环境配置

### 3. 环境隔离
- 每个项目使用独立环境
- 避免在 base 环境中安装项目包
- 使用环境变量管理配置

### 4. 版本控制
- 将环境配置文件加入版本控制
- 定期导出环境配置
- 记录环境变更原因

## 🔗 相关资源

- [Conda 官方文档](https://docs.conda.io/)
- [Anaconda 下载](https://www.anaconda.com/products/distribution)
- [Miniconda 下载](https://docs.conda.io/en/latest/miniconda.html)
- [Conda-forge 频道](https://conda-forge.org/)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
