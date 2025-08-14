# QA Clean - 中文问答净化与同义合并流水线

无分词的高精准中文问答数据清洗与同义合并流水线，基于FAISS向量召回和交叉编码器精排。

## ✨ 特性

- 🔍 **无分词设计**: 字符级n-gram处理，避免分词误差
- 🎯 **高精准合并**: 三嵌入一致性过滤 + 多CE精排融合
- 💾 **仅FAISS存储**: 支持flat/IVF/HNSW索引，GPU/CPU自适应
- 🚀 **高性能**: GPU加速的向量搜索和聚类算法
- 🐍 **Python原生**: 纯Python实现，支持Python 3.11+
- 📊 **完整观测**: JSON统计和可选图表，全流程可追踪
- 🔧 **容错设计**: 模型回退、GPU降级、自动异常处理

## 🚀 快速开始

### 前置要求

- **Anaconda** 或 **Miniconda** (推荐)
- **Python 3.11+**
- **CUDA支持** (可选，用于GPU加速)
- **输入数据**: 支持Parquet、Excel(xlsx/xls)、CSV格式，包含id/question/answer三列

### 重要说明

**FAISS安装**: FAISS GPU/CPU 必须通过 conda-forge 安装，不能通过pip安装：
```bash
# GPU版本（推荐，需要CUDA支持）
conda install -c conda-forge faiss-gpu

# CPU版本（如果没有GPU）
conda install -c conda-forge faiss-cpu
```

### 安装

#### 方法1: 使用conda（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd qa-clean

# 创建并激活环境（CPU版本）
conda env create -f environment-cpu.yml
conda activate qa-clean-cpu

# GPU版本
conda env create -f environment.yml  # 包含faiss-gpu
conda activate qa-clean
```

#### 方法2: 手动安装（新流水线）

```bash
# CPU版本
conda create -n qa-clean-pipe python=3.11 -y
conda activate qa-clean-pipe
conda install -c conda-forge faiss-cpu numpy pandas pyarrow openpyxl scikit-learn tqdm pyyaml regex -y
pip install sentence-transformers torch matplotlib rapidfuzz

# GPU版本
conda create -n qa-clean-pipe python=3.11 -y
conda activate qa-clean-pipe
conda install -c conda-forge faiss-gpu numpy pandas pyarrow openpyxl scikit-learn tqdm pyyaml regex -y
pip install sentence-transformers torch matplotlib rapidfuzz
```

### 基本用法

```bash
# 激活环境
conda activate qa-clean-pipe

# 准备数据
# 将数据放入 data/raw/ 目录，支持格式：
# - input.parquet (推荐，性能最佳)
# - input.xlsx / input.xls (Excel格式)
# - input.csv (CSV格式，自动检测编码和分隔符)
# 必须包含列：id, question, answer

# 一键运行流水线
make run
# 或
python src/run_all.py

# 分阶段运行
make stage1  # 字符级预处理与过滤
make stage2  # 三路嵌入+FAISS召回+字符n-gram补召
make stage3  # 多交叉编码器融合精排
make stage4  # 图聚类+中心约束+二次聚合
make stage5  # 答案治理与融合

# 查看结果
ls outputs/  # 所有产物：.npy, .parquet, .json, 图表
```

## 📁 输入格式

支持多种数据格式，必须包含以下三列：

- `id`: 唯一标识符
- `question`: 问题文本
- `answer`: 答案文本

### 支持的文件格式

1. **Parquet格式** (推荐，性能最佳)
   - 文件名：`input.parquet`
   - 优势：压缩率高，读取速度快

2. **Excel格式**
   - 文件名：`input.xlsx` 或 `input.xls`
   - 支持标准Excel文件

3. **CSV格式**
   - 文件名：`input.csv`
   - 自动检测编码：UTF-8、GBK、GB2312
   - 自动检测分隔符：逗号、制表符、分号

### 数据示例

```python
import pandas as pd

# 创建示例数据
df = pd.DataFrame({
    'id': [1, 2, 3],
    'question': ['如何安装Python?', 'Python怎么安装?', '什么是机器学习？'],
    'answer': ['Python可以通过官网...', '访问python.org...', '机器学习是一种...']
})

# 保存为不同格式
df.to_parquet('data/raw/input.parquet', index=False)  # Parquet格式
df.to_excel('data/raw/input.xlsx', index=False)       # Excel格式
df.to_csv('data/raw/input.csv', index=False)          # CSV格式
```

## 🏗️ 架构设计

### 五阶段流水线

1. **Stage1 - 字符级过滤**: 数据加载、Unicode标准化、规则过滤
2. **Stage2 - 三嵌入召回**: 多模型向量化、FAISS索引、n-gram补召
3. **Stage3 - CE精排**: 多交叉编码器、分数融合、分层阈值
4. **Stage4 - 图聚类**: 连通分量、中心约束、二次聚合
5. **Stage5 - 答案治理**: 冲突检测、答案融合、最终输出

### FAISS索引类型

- **flat_ip**: 暴力搜索，最高精度，适合<10万条
- **ivf_flat_ip**: IVF索引，平衡精度速度，适合10万-100万条  
- **hnsw_ip**: HNSW图索引，快速近似，适合>100万条
- **自动降级**: GPU不可用时自动切换CPU
- **模型回退**: 大模型失败时回退base模型

## 🔧 配置

### 配置文件

主配置文件：`src/configs/config.yaml`

关键参数：
- `embeddings.models`: 三路嵌入模型配置
- `recall.topk`: FAISS召回TopK
- `consistency.cos_*`: 三嵌入一致性阈值
- `rerank.thresholds`: CE精排分层阈值
- `cluster.center_constraints`: 聚类中心约束

详细配置说明见配置文件内注释和 `configs/config_variants.md`。

## 📊 输出结果

### 产物文件

- **candidate_pairs.npy**: 候选对索引数组
- **candidate_pairs_meta.parquet**: 候选对元信息
- **pair_scores.parquet**: CE精排分数
- **clusters.parquet**: 聚类结果
- **clean_answers.parquet**: 最终清洗结果
- **stage_stats.json**: 各阶段统计
- **outputs/figs/**: 可选分布图表

### 统计信息

每个阶段的处理统计、耗时、精度指标都记录在 `stage_stats.json` 中，支持全流程观测和调优。

## 🛠️ 开发

### 项目结构

```
src/
├── configs/config.yaml  # 主配置文件
├── run_all.py          # 一键运行脚本
├── recall/             # 召回模块
│   ├── base.py
│   └── faiss_provider.py
├── stages/             # 五阶段实现
│   ├── stage1_filter.py
│   ├── stage2_recall.py
│   ├── stage3_rerank.py
│   ├── stage4_cluster.py
│   └── stage5_answer_govern.py
└── utils/              # 工具模块
    ├── config.py
    ├── cn_text.py
    ├── text_sim.py
    ├── io_utils.py
    └── metrics.py
```

### 本地开发

```bash
# 克隆仓库
git clone <repository-url>
cd qa-clean

# 创建开发环境（包含开发工具）
conda env create -f environment-dev.yml
conda activate qa-clean-dev

# 运行依赖检查
make check

# 运行单元测试
make test

# 运行特定模块测试
make test-utils      # 工具模块测试
make test-stages     # 阶段模块测试
make test-cn-text    # 中文文本处理测试

# 代码格式化
ruff format .

# 类型检查
mypy src/
```

## 📝 依赖

### 核心依赖
- `pandas>=2.1.0`: 数据处理
- `sentence-transformers>=2.2.2`: 文本嵌入和交叉编码器
- `torch>=2.1.0`: PyTorch深度学习框架
- `faiss-gpu/faiss-cpu`: FAISS向量搜索（必须conda-forge安装）
- `scikit-learn>=1.3.2`: 机器学习算法
- `numpy>=1.24.0`: 数值计算
- `pyarrow`: Parquet文件读写
- `pyyaml>=6.0`: 配置文件解析
- `regex>=2023.10.3`: 高级正则表达式

### 可选依赖
- `matplotlib`: 分布图表
- `rapidfuzz`: 快速字符串相似度

### 开发依赖
- `ruff>=0.1.6`: 代码检查和格式化
- `mypy>=1.7.1`: 类型检查

## 🧪 测试

项目包含完整的单元测试覆盖核心功能：

### 测试模块

- **工具模块测试** (`tests/utils/`):
  - `test_cn_text.py`: 中文文本处理、规范化、过滤逻辑
  - `test_io_utils.py`: 文件读写、多格式支持、错误处理
  - `test_config.py`: 配置加载、验证、路径解析
  - `test_text_sim.py`: 文本相似度、n-gram、编辑距离

- **阶段模块测试** (`tests/stages/`):
  - `test_stage1_filter.py`: 数据加载、文本过滤、统计生成

### 运行测试

```bash
# 运行所有测试
make test

# 按类别运行
make test-utils      # 工具模块测试
make test-stages     # 阶段模块测试

# 运行特定模块
make test-cn-text    # 中文文本处理
make test-io         # IO工具
make test-config     # 配置管理
make test-sim        # 文本相似度
make test-stage1     # Stage1过滤

# 其他选项
make test-verbose    # 详细输出
make test-quiet      # 静默模式
make test-list       # 列出所有测试用例
```

### 测试特性

- **Mock支持**: 使用unittest.mock模拟外部依赖
- **边界测试**: 覆盖边界情况和错误处理
- **中文支持**: 专门测试中文文本处理逻辑
- **格式兼容**: 测试多种文件格式读取
- **彩色输出**: 清晰的测试结果显示

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
  raw/README.md       # 数据格式说明（请用自己数据替换）
outputs/
  figs/
```

### 依赖安装（CPU/GPU）
- CPU：
```bash
conda create -n qa-clean-pipe python=3.11 -y
conda activate qa-clean-pipe
conda install -c conda-forge faiss-cpu numpy pandas pyarrow openpyxl scikit-learn tqdm pyyaml regex -y
pip install sentence-transformers torch
# 可选
pip install matplotlib rapidfuzz
```
- GPU：
```bash
conda create -n qa-clean-pipe python=3.11 -y
conda activate qa-clean-pipe
conda install -c conda-forge faiss-gpu numpy pandas pyarrow openpyxl scikit-learn tqdm pyyaml regex -y
pip install sentence-transformers torch
# 可选
pip install matplotlib rapidfuzz
```

### 数据准备
- 支持多种格式：`input.parquet` (推荐) / `input.xlsx` / `input.csv`
- 必含三列：`id`, `question`, `answer`
- CSV文件自动检测编码和分隔符

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
