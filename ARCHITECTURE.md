# QA清洗流水线架构优化说明

## 📋 优化概述

本次架构优化解决了原有系统的多个关键问题，提升了代码质量、可维护性和用户体验。

## 🔧 主要改进

### 1. 统一日志系统 (`src/utils/logger.py`)

**问题**: 原有日志分散、格式不统一、缺乏进度追踪
**解决方案**: 
- ✅ 结构化日志记录 (支持JSON输出)
- ✅ 多级别日志 (DEBUG/INFO/WARNING/ERROR)  
- ✅ 线程安全的进度追踪
- ✅ 性能计时器与资源监控
- ✅ 灵活的输出目标 (控制台/文件)

**使用示例**:
```python
from src.utils.logger import setup_logger, get_logger

# 设置日志器
logger = setup_logger(log_file="logs/pipeline.log", level=LogLevel.INFO)

# 使用阶段计时器
with logger.stage_timer("stage1", "数据预处理"):
    # 执行阶段代码
    pass

# 使用操作计时器
with logger.operation_timer("stage1", "数据加载"):
    # 执行操作代码
    pass
```

### 2. 环境管理模块 (`src/utils/environment.py`)

**问题**: 缺乏自动环境检测、GPU/CPU配置混乱、依赖管理不完善
**解决方案**:
- ✅ 自动GPU/CPU设备检测与配置推荐
- ✅ 依赖库可用性检查与回退机制
- ✅ 环境变量自动管理
- ✅ 性能优化的批次大小推荐
- ✅ 完整的环境信息摘要

**特性**:
```python
from src.utils.environment import get_env_manager

env = get_env_manager()

# 获取系统信息
sys_info = env.get_system_info()
print(f"GPU可用: {sys_info.cuda_available}")
print(f"GPU内存: {sys_info.gpu_memory}")

# 获取推荐配置
device = env.get_recommended_device()  # 'cuda' 或 'cpu'
batch_size = env.get_optimal_batch_size('embedding')  # 基于GPU内存优化

# 环境验证
is_valid, errors = env.validate_environment()
```

### 3. 增强配置管理 (`src/utils/config.py`)

**问题**: 配置静态、缺乏环境适应性、不支持运行时调整
**解决方案**:
- ✅ 环境变量集成 (支持 `QA_DEVICE`, `QA_BATCH_SIZE` 等)
- ✅ 运行时配置优化应用
- ✅ 智能默认值管理
- ✅ 配置路径支持 (点号分隔)

**环境变量支持**:
```bash
export QA_DEVICE=cuda          # 覆盖设备配置
export QA_BATCH_SIZE=64        # 覆盖批次大小
export QA_OUTPUT_DIR=./results # 覆盖输出目录
```

### 4. 重构启动机制 (`src/pipeline.py` + `run.py`)

**问题**: 模块导入错误、启动复杂、错误处理不优雅
**解决方案**:
- ✅ 自动路径管理，解决导入问题
- ✅ 优雅的错误处理和状态报告
- ✅ 模块化的阶段管理
- ✅ 灵活的阶段选择 (支持 `--start-stage` / `--end-stage`)

## 🚀 新的使用方式

### 基本使用

```bash
# 方式1: 使用简化启动脚本 (推荐)
python run.py -i data/raw/qa.xlsx

# 方式2: 直接使用pipeline模块
python src/pipeline.py -i data/raw/qa.xlsx

# 方式3: 通过conda环境
conda run -n qa-clean python run.py -i data/raw/qa.xlsx
```

### 高级使用

```bash
# 仅运行特定阶段
python run.py -i data/raw/qa.xlsx --start-stage stage2 --end-stage stage4

# 指定输出目录
python run.py -i data/raw/qa.xlsx -o /path/to/output

# 调整日志级别
python run.py -i data/raw/qa.xlsx --log-level DEBUG

# 使用环境变量优化
export QA_DEVICE=cpu QA_BATCH_SIZE=32
python run.py -i data/raw/qa.xlsx
```

## 📊 日志输出改进

### 之前的日志
```
[stage2] 加载 30000 个相似对
[stage3] CE加载失败：xxx，尝试回退基础模型
```

### 现在的日志
```
[2025-08-18 15:50:54.624] [INFO] [pipeline] ============================================================
[2025-08-18 15:50:54.624] [INFO] [pipeline] QA清洗流水线启动
[2025-08-18 15:50:54.624] [INFO] [pipeline] ============================================================
[2025-08-18 15:50:54.624] [INFO] [pipeline] 开始环境验证
[2025-08-18 15:51:00.700] [INFO] [pipeline] Python版本: 3.11.13
[2025-08-18 15:51:00.700] [INFO] [pipeline] 平台: Linux
[2025-08-18 15:51:00.700] [INFO] [pipeline] CPU核心数: 192
[2025-08-18 15:51:00.700] [INFO] [pipeline] GPU可用: 1 块
[2025-08-18 15:51:00.701] [INFO] [pipeline] GPU0 内存: 23.6GB
[2025-08-18 15:51:00.701] [INFO] [pipeline] 推荐设备: cuda
[2025-08-18 15:51:00.701] [INFO] [pipeline] 推荐嵌入批次: 128
[2025-08-18 15:51:00.701] [INFO] [pipeline] 推荐精排批次: 64
[2025-08-18 15:51:02.209] [INFO] [stage1] 字符级预处理与过滤 完成 (耗时: 1.50s)
```

## 🎯 性能优化

1. **自动批次大小调整**: 基于GPU内存自动优化批次大小
2. **设备自动选择**: 智能检测并选择最优设备
3. **并行度优化**: 基于CPU核心数自动调整并行度
4. **内存监控**: 实时监控GPU/CPU内存使用

## 🔒 错误处理改进

1. **依赖缺失**: 自动检测并提供安装建议
2. **GPU不可用**: 自动回退到CPU路径
3. **配置错误**: 详细的验证错误信息
4. **运行时异常**: 优雅的错误报告和清理

## 📁 文件结构

```
src/
├── pipeline.py           # 主流水线入口 (新)
├── utils/
│   ├── logger.py         # 统一日志系统 (新)
│   ├── environment.py    # 环境管理 (新)
│   ├── config.py         # 增强配置管理 (更新)
│   └── metrics.py        # 原有指标系统 (更新)
├── stages/               # 各阶段模块 (更新日志)
└── configs/              # 配置文件
run.py                    # 简化启动脚本 (新)
```

## 🆚 对比总结

| 方面 | 优化前 | 优化后 |
|------|--------|--------|
| **启动方式** | 复杂的路径设置 | 一键启动 `python run.py` |
| **日志系统** | 分散、格式不统一 | 结构化、带时间戳、进度追踪 |
| **环境管理** | 手动配置 | 自动检测、智能推荐 |
| **错误处理** | 粗糙的异常信息 | 优雅的错误报告和恢复 |
| **配置管理** | 静态配置文件 | 动态+环境变量集成 |
| **性能优化** | 固定参数 | 基于硬件自动调优 |

这次架构优化极大提升了系统的易用性、可靠性和性能，为后续开发和维护奠定了坚实基础。
