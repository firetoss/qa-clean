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
- **输入数据**: 支持Parquet、Excel(xlsx/xls)、CSV格式，仅需question/answer两列（id列可选）

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
# 必须包含列：question, answer（id列可选，无则自动创建）

# 一键运行流水线（使用配置文件中的输入路径）
make run
# 或
python src/run_all.py --config src/configs/config.yaml

# 指定输入文件运行（推荐）
make run INPUT=./data/raw/qa.xlsx
# 或
python src/run_all.py --config src/configs/config.yaml --input ./data/raw/qa.xlsx

# 分阶段运行（使用配置文件中的输入路径）
make stage1  # 字符级预处理与过滤
make stage2  # 三路嵌入+FAISS召回+字符n-gram补召
make stage3  # 多交叉编码器融合精排
make stage4  # 图聚类+中心约束+二次聚合
make stage5  # 答案治理与融合

# 分阶段运行（指定输入文件）
make stage1 INPUT=./data/raw/qa.xlsx
make stage2 INPUT=./data/raw/qa.xlsx
# ... 其他阶段类似

# 查看结果
ls outputs/  # 所有产物：.npy, .parquet, .json, 图表
```

## 📁 输入格式

支持多种数据格式，必须包含以下列：

- `question`: 问题文本（必需）
- `answer`: 答案文本（必需）
- `id`: 唯一标识符（可选，无则自动创建行索引）

### 命令行参数说明

**方式1：使用配置文件中的输入路径**
```bash
# 修改 src/configs/config.yaml 中的 input_path
python src/run_all.py --config src/configs/config.yaml
```

**方式2：直接指定输入文件（推荐）**
```bash
# 直接指定输入文件，无需修改配置文件
python src/run_all.py --config src/configs/config.yaml --input ./data/raw/qa.xlsx
```

**使用Makefile**
```bash
# 使用配置文件中的路径
make run

# 指定输入文件
make run INPUT=./data/raw/qa.xlsx
```

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

# 创建示例数据（id列可选）
df = pd.DataFrame({
    'question': ['如何安装Python?', 'Python怎么安装?', '什么是机器学习？'],
    'answer': ['Python可以通过官网...', '访问python.org...', '机器学习是一种...']
})

# 可选：添加id列
# df['id'] = [1, 2, 3]  # 如果不添加，程序会自动创建

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
4. **Stage4 - 图聚类**: 多引擎聚类、中心约束、二次聚合
5. **Stage5 - 答案治理**: 冲突检测、答案融合、最终输出

### Stage4 聚类架构

Stage4 采用统一接口设计，支持三种聚类引擎的无缝切换：

```
stage4_cluster.py (统一入口)
├── stage4_cluster_networkx.py (NetworkX引擎)
├── stage4_cluster_parallel.py (并行引擎)  
└── stage4_cluster_original.py (原始引擎)
```

**架构特点：**
- **统一接口**：所有引擎使用相同的函数签名和配置格式
- **自动回退**：NetworkX引擎依赖缺失时自动回退到并行引擎
- **配置驱动**：通过配置文件选择引擎，无需修改代码
- **性能优化**：每个引擎针对特定场景进行深度优化
- **代码规范**：统一的代码风格、注释和文档

**引擎协作流程：**
1. 统一入口根据配置加载指定引擎
2. 依赖检测和自动回退机制
3. 引擎执行聚类算法
4. 统一的输出格式和统计信息
5. 一致的错误处理和日志记录

### FAISS索引类型

- **flat_ip**: 暴力搜索，最高精度，适合<10万条
- **ivf_flat_ip**: IVF索引，平衡精度速度，适合10万-100万条  
- **hnsw_ip**: HNSW图索引，快速近似，适合>100万条
- **自动降级**: GPU不可用时自动切换CPU
- **模型回退**: 大模型失败时回退base模型

## 💡 核心算法逻辑

### 📋 数据流转详解

```
原始QA数据 → Stage1过滤 → Stage2召回 → Stage3重排 → Stage4聚类 → Stage5治理 → 最终结果
     ↓           ↓          ↓          ↓          ↓          ↓
   20K条    →  19.9K条  →  1.1M对  →  1.1M对  →   273簇   →   273簇
```

### 🎯 聚类与冲突处理

#### 聚类结构说明
- **cluster_id**: 聚类编号（非原始问题ID）
- **members**: 属于该聚类的原始问题ID列表
- **center**: 聚类中心问题的ID（图中度数最高的节点）
- **center_question**: 聚类中心对应的问题文本

#### 🔍 智能冲突检测

通过语义信号分析自动检测答案冲突：

```python
signals = extract_signals(answer)  # 提取关键信息
# 包含：金额(money)、日期(date)、否定词(neg)、条件(cond)等

def hard_conflict(signal_a, signal_b):
    # 金额冲突: 数值差异 > tolerance_pct (默认5%)
    # 否定冲突: 肯定 vs 否定表述  
    # 日期冲突: 日期相差 > tolerance_days (默认1天)
```

#### 📝 答案处理策略

| 冲突状态 | 处理方式 | 输出结果 |
|---------|---------|---------|
| **无冲突** | 智能融合所有答案 | `merged_answer` = "答案1；答案2；答案3" |
| **有冲突** | 选择最频繁答案 | `merged_answer` = `representative_answer` |

**实际示例：**
```yaml
# 无冲突情况
聚类成员答案:
  - "需要身份证原件"
  - "需要身份证和户口本"
融合结果: "需要身份证原件；需要身份证和户口本"

# 有冲突情况  
聚类成员答案:
  - "办理费用100元"
  - "办理费用200元"  
处理结果: "办理费用100元" (选择更频繁的答案)
```

### 🏷️ 多格式输出

同时生成3种格式的最终结果：
- **📄 Parquet**: 保持原始数据结构（members为列表），便于程序处理
- **📊 CSV**: members转为逗号分隔字符串，便于Excel查看
- **📈 Excel**: 直接可用的表格格式，便于业务人员使用

## 🔧 配置

### 配置文件

主配置文件：`src/configs/config.yaml`

关键参数：
- `embeddings.models`: 三路嵌入模型配置
- `recall.topk`: FAISS召回TopK
- `consistency.cos_*`: 三嵌入一致性阈值
- `rerank.thresholds`: CE精排分层阈值
- `cluster.center_constraints`: 聚类中心约束
- `cluster.use_parallel`: 是否启用多核并行聚类（推荐开启）
- `cluster.n_jobs`: 并行进程数（-1为使用所有CPU核心）
 - `cluster.enable_gpu`: 是否启用GPU图聚类（需安装cudf/cugraph）

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

## ⚡ 性能优化

### 三种聚类引擎

Stage4聚类阶段提供三种可选的聚类引擎，经过全面优化，支持不同场景需求：

#### 🥇 NetworkX引擎 (推荐 - 最高质量)

基于NetworkX库的高级社区检测算法，提供最佳聚类质量：

```yaml
# 在 src/configs/config.yaml 中配置
cluster:
  engine: "networkx"    # 使用NetworkX引擎（默认GPU加速，自动回退CPU）
  method: "leiden"      # 支持leiden/louvain/connected_components
  enable_gpu: true      # 默认开启；若无cugraph/cudf则自动回退CPU
  use_parallel: true    # CPU回退时仍可并行
  n_jobs: -1           # 使用所有CPU核心
  resolution: 1.0      # 聚类分辨率（越大簇越小）
```

**算法优势：**
- 🎯 **Leiden算法**：最先进的社区检测，克服Louvain算法局限
- 📊 **模块度优化**：基于图论的严格数学基础
- 🔧 **分辨率调节**：精确控制聚类粒度和簇数量
- ⚖️ **加权处理**：充分利用CE分数权重信息

**特点：**
- ✅ 最高质量社区发现，显著优于连通分量
- ✅ 算法丰富：Leiden、Louvain、连通分量
- ✅ 自动依赖回退：缺失时回退到并行引擎
- ✅ 并行优化：大图分块并行处理
- ⚠️ 内存需求较高，适合大内存环境

**依赖安装（CPU）：**
```bash
# 必需依赖
pip install networkx

# 可选依赖（用于Leiden算法）
pip install python-igraph leidenalg
```

**GPU 加速（可选，需 NVIDIA GPU + RAPIDS）：**
```bash
# 🚀 GPU图聚类加速 - 10-100x性能提升
# 安装 RAPIDS cuGraph/cudf (根据CUDA版本选择)
conda install -c rapidsai -c conda-forge cugraph>=23.10 cudf>=23.10

# RAPIDS要求：
# - NVIDIA GPU (Compute Capability >= 6.0) 
# - CUDA 11.8+ 或 12.0+
# - GPU内存 >= 8GB (推荐)

# 配置文件中启用GPU:
# src/configs/config.yaml
# cluster:
#   engine: "networkx"
#   enable_gpu: true
```

**安装验证：**
```bash
# 验证GPU环境
python -c "import cudf, cugraph; print('✅ RAPIDS GPU加速可用')"

# 性能测试
python benchmark_clustering_engines.py  # 对比GPU/CPU性能
```

#### ⚡ Parallel引擎 (高性能 - 无额外依赖)

多核并行优化的连通分量算法，内存友好的高性能实现：

```yaml
cluster:
  engine: "parallel"    # 使用并行引擎
  use_parallel: true    # 启用并行计算
  n_jobs: -1           # 使用所有CPU核心
```

**性能优化：**
- 🚀 **并行计算**：多进程并行连通分量检测
- 🧠 **智能切换**：根据数据规模自动选择串行/并行
- 💾 **内存优化**：智能数据分块和内存管理
- ⚖️ **负载均衡**：动态调整工作负载分配

**特点：**
- ✅ 高性能：2-8x性能提升
- ✅ 无额外依赖：仅使用标准库
- ✅ 内存友好：适合内存受限环境
- ✅ 自适应：智能算法选择
- ⚠️ 仅支持连通分量聚类

**性能数据：**
- 小数据集（<1K节点）：串行模式，避免并行开销
- 中等数据集（1K-100K节点）：2-4x加速
- 大数据集（>100K节点）：4-8x加速

#### 🔧 Original引擎 (最小依赖 - 最高兼容性)

原始单核连通分量算法，提供最大兼容性：

```yaml
cluster:
  engine: "original"    # 使用原始引擎
```

**特点：**
- ✅ 最小依赖：仅需标准库和基础科学计算包
- ✅ 高兼容性：适用于各种环境和平台
- ✅ 调试友好：代码简洁清晰，易于理解
- ✅ 稳定可靠：经过充分测试和验证
- ⚠️ 性能较低：适合小规模数据集（<10万节点）

#### 📊 引擎选择指南

| 使用场景 | 推荐引擎 | 理由 |
|---------|---------|------|
| **研究/生产环境** | `networkx` | 最高质量聚类，算法先进 |
| **大规模数据** | `networkx` | Leiden算法处理大图优势明显 |
| **性能优先** | `parallel` | 高性能，无额外依赖 |
| **内存受限** | `parallel` | 内存友好，优化的数据结构 |
| **小规模数据** | `original` | 简单高效，无并行开销 |
| **兼容性优先** | `original` | 最小依赖，最高兼容性 |
| **调试开发** | `original` | 代码简洁，易于理解和修改 |

#### 🛠️ 引擎特性对比

| 特性 | NetworkX | Parallel | Original |
|------|----------|----------|----------|
| **聚类质量** | 🥇 最高 | 🥈 中等 | 🥈 中等 |
| **性能速度** | 🥈 中等 | 🥇 最快 | 🥉 较慢 |
| **内存使用** | 🥉 较高 | 🥈 中等 | 🥇 最低 |
| **依赖要求** | 📦 需额外依赖 | 🎯 无额外依赖 | 🎯 无额外依赖 |
| **算法丰富度** | 🎨 丰富 | 🔧 基础 | 🔧 基础 |
| **并行支持** | ✅ 支持 | ✅ 支持 | ❌ 不支持 |
| **自动回退** | ✅ 支持 | ❌ 不适用 | ❌ 不适用 |

### 🚀 GPU图聚类加速 

NetworkX引擎支持RAPIDS cuGraph GPU加速，在大规模数据上提供显著性能提升：

#### ✨ 核心特性

- **🔥 GPU优先**：自动检测并优先使用RAPIDS cuGraph
- **🔄 智能回退**：GPU失败时无缝回退到CPU NetworkX
- **⚡ 高性能**：大图处理10-100x性能提升
- **🎯 高质量**：保持与CPU相同的聚类质量
- **💾 内存优化**：GPU VRAM智能管理

#### 📊 算法支持

| 算法 | CPU实现 | GPU实现 | 特点 |
|------|---------|---------|------|
| **Leiden** | igraph + leidenalg | cuGraph | 最先进社区检测，质量最高 |
| **Louvain** | NetworkX内置 | cuGraph | 经典模块度优化，速度快 |
| **连通分量** | NetworkX内置 | cuGraph | 基础图连通性分析 |

#### 🚀 性能表现

实际测试数据（17K节点，1M+边）：
- **GPU模块度**: 0.8077+（高质量聚类）
- **聚类数量**: 273个
- **平均聚类大小**: 64.7
- **GPU加速效果**: 大规模图显著提升

#### 使用示例

```yaml
# 启用GPU加速配置
cluster:
  engine: "networkx"        # 使用NetworkX引擎
  method: "leiden"          # 推荐使用Leiden算法
  enable_gpu: true          # 🚀 启用GPU加速
  resolution: 1.0
  use_parallel: true        # CPU回退时的并行配置
```

#### 系统要求

- **GPU**: NVIDIA GPU (Compute Capability >= 6.0)
- **CUDA**: 11.8+ 或 12.0+
- **内存**: GPU VRAM >= 8GB (推荐)
- **驱动**: 最新NVIDIA驱动
- **依赖**: RAPIDS cuGraph/cudf

#### 自动回退机制

GPU不可用时自动回退到CPU实现：
1. **检测GPU环境**: 自动检测cuGraph/cudf可用性
2. **智能回退**: GPU失败时无缝切换到CPU NetworkX
3. **统一接口**: 相同输入输出格式
4. **错误处理**: 详细的错误信息和建议

### 性能测试

项目包含内置的聚类引擎性能对比工具：

```bash
# 运行三引擎性能对比测试
python benchmark_clustering_engines.py

# 运行聚类算法对比测试  
python benchmark_clustering.py
```

**测试内容：**
- 自动生成测试数据（可配置规模）
- 三引擎并行测试和性能对比
- 详细的性能报告和推荐
- 失败引擎的依赖提示

**测试输出示例：**
```
🚀 三种聚类引擎性能对比测试
==========================================
创建测试数据完成：18743 个相似对，1982 个节点
CPU核心数: 8

🧪 测试 ORIGINAL 引擎...
✅ ORIGINAL 引擎完成，耗时: 2.34秒

🧪 测试 PARALLEL 引擎...  
✅ PARALLEL 引擎完成，耗时: 0.87秒

🧪 测试 NETWORKX 引擎...
✅ NETWORKX 引擎完成，耗时: 1.23秒

📈 性能对比结果:
==========================================
排名 | 引擎      | 耗时(秒) | 相对性能
-----------------------------------------
 1   | parallel  |    0.87 | 1.00x
 2   | networkx  |    1.23 | 0.71x  
 3   | original  |    2.34 | 0.37x

🔍 详细分析:
最快引擎: PARALLEL
最慢引擎: ORIGINAL  
性能差异: 2.69x

💡 推荐方案:
🥇 NETWORKX: 功能最强，支持多种高级聚类算法
⚡ PARALLEL: 性能优化，无额外依赖
🔧 ORIGINAL: 最小依赖，调试友好
```

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
- 必含列：`question`, `answer`（`id`列可选，无则自动创建）
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
