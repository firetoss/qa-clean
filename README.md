# QA Clean - QA 数据清洗与治理工具

一个强大的QA数据清洗与治理工具，支持聚合去重、聚类合并、代表问题输出等功能。

## ✨ 特性

- 🔍 **智能去重**: 基于双嵌入模型的语义相似度检测
- 🎯 **聚类合并**: 自动识别相似问题并生成代表问题
- 💾 **多存储支持**: 支持FAISS GPU和PostgreSQL+pgvector
- 🚀 **高性能**: GPU加速的向量搜索和聚类算法
- 🐍 **Python原生**: 纯Python实现，支持Python 3.11+
- 📊 **灵活输出**: 支持CSV、Excel等多种格式
- 🔧 **NumPy 2.x兼容**: 使用FAISS 1.12.0，完全支持NumPy 2.x

## 🚀 快速开始

### 前置要求

- **Anaconda** 或 **Miniconda** (推荐)
- **Python 3.11+**
- **CUDA支持** (可选，用于GPU加速)
- **NumPy 2.x支持** (FAISS 1.12.0完全兼容)

### 重要说明

**FAISS安装**: FAISS GPU/CPU 必须通过 conda-forge 安装，不能通过pip安装：
```bash
# GPU版本（推荐，需要CUDA支持）
conda install -c conda-forge faiss-gpu>=1.12.0

# CPU版本（如果没有GPU）
conda install -c conda-forge faiss-cpu>=1.12.0
```

### 安装

#### 方法1: 使用conda（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd qa-clean

# 创建并激活环境
conda env create -f environment.yml
conda activate qa-clean

# 安装项目
pip install -e .
```



#### 方法2: 手动安装

```bash
# 创建环境
conda create -n qa-clean python=3.11

# 激活环境
conda activate qa-clean

# 安装依赖
conda install -c conda-forge pandas>=2.1.0 openpyxl>=3.1.2 scikit-learn>=1.3.2 numpy>=1.24.0 tqdm>=4.66.0 jieba>=0.42.1
conda install -c pytorch pytorch>=2.1.0

# 安装FAISS（必须通过conda-forge）
conda install -c conda-forge faiss-gpu>=1.12.0

# 安装其他依赖
pip install sentence-transformers>=2.2.2 psycopg2-binary>=2.9.9

# 安装项目
pip install -e .
```

### 开发环境

```bash
# 创建开发环境
conda env create -f environment-dev.yml
conda activate qa-clean-dev

# 安装项目
pip install -e ".[dev]"
```

### 基本用法

```bash
# 激活环境
conda activate qa-clean

# 处理QA数据（使用FAISS GPU，默认）
qa-clean process data.csv --output results.csv

# 使用PostgreSQL+pgvector存储
qa-clean process data.csv --vector-store pgvector --output results.csv

# 搜索相似问题
qa-clean search "如何安装Python?" --vector-store faiss_gpu

# 查看存储信息
qa-clean info --vector-store faiss_gpu
```

## 📁 输入格式

支持CSV和Excel格式，需要包含以下列：

- `question`: 问题文本
- `answer`: 答案文本
- 其他列将作为元数据保存

示例：
```csv
question,answer,category,source
如何安装Python?,Python可以通过官网下载安装包...,技术,官方文档
Python怎么安装?,访问python.org下载安装程序...,技术,用户手册
```

## 🏗️ 架构设计

### 向量存储选项

#### 1. FAISS GPU (faiss_gpu) - 默认推荐
- **优势**: 高性能向量搜索，GPU加速，速度极快，无需外部数据库，支持Python 3.11+，完全支持NumPy 2.x
- **劣势**: 数据不持久化，内存占用较高，重启后数据丢失，需要GPU资源
- **适用场景**: 开发测试、高性能要求、快速原型、GPU环境、NumPy 2.x环境

#### 2. PostgreSQL + pgvector (pgvector)
- **优势**: 企业级稳定性，支持复杂查询，可扩展性强，支持事务和ACID，数据持久化
- **劣势**: 需要PostgreSQL环境，部署相对复杂，资源消耗较高
- **适用场景**: 大规模生产、企业环境、需要复杂查询

### 推荐选择

- **开发/测试环境**: 使用 `faiss_gpu` 存储
- **生产环境**: 有PostgreSQL时使用 `pgvector`，否则使用 `faiss_gpu`
- **快速原型**: 使用 `faiss_gpu` 存储
- **NumPy 2.x环境**: 推荐使用 `faiss_gpu`（FAISS 1.12.0）

## 🔧 配置

### 环境变量

```bash
# PostgreSQL配置（仅pgvector存储需要）
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=qa_clean
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=password
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vector_store` | faiss_gpu | 向量存储类型 (faiss_gpu/pgvector) |
| `gpu_id` | 0 | GPU设备ID (仅faiss_gpu) |
| `topk` | 100 | 相似度搜索top-k值 |
| `question_col` | question | 问题列名 |
| `answer_col` | answer | 答案列名 |

## 📊 输出结果

处理完成后会生成以下信息：

- **原始数据数量**: 输入的QA对总数
- **去重后数量**: 去重后的QA对数量
- **聚类数量**: 识别出的相似问题聚类数
- **代表问题**: 每个聚类的代表性问题

输出文件包含：
- `id`: 唯一标识符
- `question`: 问题文本
- `answer`: 答案文本
- `cluster_id`: 聚类ID
- `representative_question`: 代表性问题
- `metadata`: 其他元数据

## 🛠️ 开发

### 项目结构

```
src/qa_clean/
├── __init__.py
├── cli.py              # 命令行接口
├── processor.py        # 核心处理器
├── models.py           # 双嵌入模型管理
├── vector_factory.py   # 向量存储工厂
├── faiss_store.py      # FAISS GPU存储
├── vector_store.py     # PostgreSQL+pgvector存储
├── config.py           # 配置管理
├── clustering.py       # 聚类算法
└── utils.py            # 工具函数
```

### 本地开发

```bash
# 克隆仓库
git clone <repository-url>
cd qa-clean

# 创建开发环境
conda env create -f environment-dev.yml
conda activate qa-clean-dev

# 安装项目
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
ruff format .

# 类型检查
mypy src/
```

## 📝 依赖

### 核心依赖
- `pandas>=2.1.0`: 数据处理
- `sentence-transformers>=2.2.2`: 文本嵌入模型
- `torch>=2.1.0`: PyTorch深度学习框架
- `faiss-gpu>=1.12.0`: FAISS GPU向量搜索（支持NumPy 2.x）
- `scikit-learn>=1.3.2`: 机器学习算法
- `numpy>=1.24.0`: 数值计算（支持NumPy 2.x）

### 存储依赖
- `psycopg2-binary>=2.9.9`: PostgreSQL连接器（pgvector）

### 开发依赖
- `ruff>=0.1.6`: 代码检查和格式化
- `mypy>=1.7.1`: 类型检查
- `pytest>=7.4.3`: 测试框架

## 🔧 故障排除

### 常见问题

如果遇到问题，可以尝试重建环境：

```bash
# 删除旧环境
conda env remove -n qa-clean

# 重新创建环境
conda env create -f environment.yml
conda activate qa-clean
pip install -e .
```



## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) - 中文嵌入模型
- [moka-ai/m3e-large](https://huggingface.co/moka-ai/m3e-large) - 多语言嵌入模型
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI相似性搜索库
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL向量扩展

---

## 中文问答净化与同义合并流水线（仅FAISS，无分词）

### 目录结构（新增）
```
src/
  configs/config.yaml
  run_all.py
  recall/
    base.py
    faiss_provider.py
  stages/
    stage1_filter.py
    stage2_recall.py
    stage3_rerank.py
    stage4_cluster.py
    stage5_answer_govern.py
  utils/
    config.py
    cn_text.py
    text_sim.py
    io_utils.py
    metrics.py
data/
  raw/input.parquet   # 示例占位（请用自己数据替换）
outputs/
  figs/
```

### 依赖安装（CPU/GPU）
- CPU：
```bash
conda create -n qa-clean-pipe python=3.11 -y
conda activate qa-clean-pipe
conda install -c conda-forge faiss-cpu numpy pandas pyarrow scikit-learn tqdm pyyaml regex -y
pip install sentence-transformers
# 可选
pip install matplotlib rapidfuzz
```
- GPU：
```bash
conda create -n qa-clean-pipe python=3.11 -y
conda activate qa-clean-pipe
conda install -c conda-forge faiss-gpu numpy pandas pyarrow scikit-learn tqdm pyyaml regex -y
pip install sentence-transformers
# 可选
pip install matplotlib rapidfuzz
```

### 数据准备
- `data/raw/input.parquet` 必含三列：`id`, `question`, `answer`

### 运行命令
```bash
# 逐阶段
python src/stages/stage1_filter.py
python src/stages/stage2_recall.py
python src/stages/stage3_rerank.py
python src/stages/stage4_cluster.py
python src/stages/stage5_answer_govern.py

# 一键运行
python src/run_all.py
```

### 产物
- `candidate_pairs.npy`
- `candidate_pairs_meta.parquet`
- `pair_scores.parquet`
- `clusters.parquet`
- `clean_answers.parquet`
- `stage_stats.json`

### 说明
- 流水线严格不依赖中文分词，字符级归一化与 n-gram 相似补召。
- 召回仅使用 FAISS，支持 `flat_ip / ivf_flat_ip / hnsw_ip` 并可持久化索引。
- 高精准阈值：三嵌入一致性与 CE 分层阈值按 `src/configs/config.yaml` 默认值执行，可自行调优。

### Makefile 使用
- 查看帮助：`make help`
- 一键运行（模块方式）：`make run`
- 一键运行（脚本方式）：`make pipeline`
- 分阶段：`make stage1` ~ `make stage5`
- 依赖自检：`make check`
- 清理产物：`make clean`
- 环境安装提示：`make env-cpu` / `make env-gpu`

### 配置调优
详见 `configs/config_variants.md`，包含：
- 大规模数据配置 (>100万问题)
- 低内存配置 (<8GB显存)
- 高精度配置 (质量优先)
- 性能优化配置 (速度优先)

### 故障排除

#### 常见问题与解决方案

1. **FAISS安装失败**
   ```bash
   # GPU版本
   conda install -c conda-forge faiss-gpu
   # CPU版本
   conda install -c conda-forge faiss-cpu
   ```

2. **模型下载失败**
   - 检查网络连接
   - 使用HuggingFace镜像：`export HF_ENDPOINT=https://hf-mirror.com`
   - 程序会自动回退到base模型继续运行

3. **内存不足**
   - 降低batch_size：`embeddings.batch_size: 16`
   - 使用CPU：`device: "cpu"`
   - 减少TopK：`recall.topk: 50`

4. **精度不满意**
   - 提高阈值：参考`configs/config_variants.md`高精度配置
   - 检查数据质量和过滤规则
   - 调整模型组合

5. **速度太慢**
   - 使用HNSW索引：`index_type: "hnsw_ip"`
   - 关闭二次聚合：`second_merge.enable: false`
   - 参考性能优化配置

### 日志解读
- `[stage1]` ~ `[stage5]`: 各阶段进度
- `GPU不可用，自动切换到CPU`: 正常降级
- `加载xxx失败，尝试回退基础模型`: 模型自动降级
- 统计输出在 `outputs/stage_stats.json`
