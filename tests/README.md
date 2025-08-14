# QA Clean 项目单元测试

本目录包含 QA Clean 项目的完整单元测试套件，覆盖核心功能模块和业务逻辑。

## 📁 测试结构

```
tests/
├── README.md              # 测试文档
├── __init__.py             # 测试包初始化
├── test_runner.py          # 测试运行器（彩色输出、分类运行）
├── test_config.yaml        # 测试配置文件
├── utils/                  # 工具模块测试
│   ├── __init__.py
│   ├── test_cn_text.py     # 中文文本处理测试
│   ├── test_io_utils.py    # IO工具测试
│   ├── test_config.py      # 配置管理测试
│   └── test_text_sim.py    # 文本相似度测试
└── stages/                 # 阶段模块测试
    ├── __init__.py
    └── test_stage1_filter.py # Stage1过滤测试
```

## 🧪 测试模块详述

### 工具模块测试 (`tests/utils/`)

#### 1. `test_cn_text.py` - 中文文本处理测试
- **测试范围**: 文本规范化、全角转半角、过滤规则
- **核心功能**:
  - `to_halfwidth()`: 全角字符转半角
  - `normalize_zh()`: 中文文本标准化
  - `filter_reason()`: 问题过滤逻辑
  - `match_any()`: 正则模式匹配
- **测试特色**: 
  - 中文Unicode处理
  - 标点符号映射
  - 白名单/黑名单优先级
  - 边界情况处理

#### 2. `test_io_utils.py` - IO工具测试
- **测试范围**: 文件读写、多格式支持、错误处理
- **核心功能**:
  - `read_data_file()`: 智能文件读取（parquet/xlsx/csv）
  - `save_npy()`: NumPy数组保存
  - `save_json_merge()`: JSON合并保存
  - `ensure_parent_dir()`: 目录创建
- **测试特色**:
  - Mock外部依赖
  - 文件格式自动检测
  - 编码兼容性测试
  - 错误恢复机制

#### 3. `test_config.py` - 配置管理测试
- **测试范围**: 配置加载、验证、路径解析
- **核心功能**:
  - `Config.get()`: 嵌套路径访问
  - `load_config()`: YAML配置加载
  - `_validate_config()`: 配置验证
  - `ensure_seed()`: 随机种子设置
- **测试特色**:
  - 嵌套配置访问
  - 参数范围验证
  - 文件错误处理
  - 默认值机制

#### 4. `test_text_sim.py` - 文本相似度测试
- **测试范围**: 字符n-gram、相似度计算、编辑距离
- **核心功能**:
  - `char_ngrams()`: 字符级n-gram生成
  - `jaccard()`: Jaccard相似度
  - `ngram_jaccard()`: n-gram Jaccard相似度
  - `edit_distance()`: 编辑距离计算
- **测试特色**:
  - 中文字符处理
  - 数学正确性验证
  - 性能特征测试
  - 对称性验证

### 阶段模块测试 (`tests/stages/`)

#### 1. `test_stage1_filter.py` - Stage1过滤测试
- **测试范围**: 数据加载、文本过滤、统计生成
- **核心功能**:
  - `_load_or_sample()`: 数据加载逻辑
  - `run()`: 完整过滤流程
  - 文件格式支持
  - 统计信息生成
- **测试特色**:
  - Mock数据框操作
  - 多种文件格式
  - 过滤统计验证
  - 错误恢复测试

## 🚀 运行测试

### 基本运行命令

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
```

### 高级选项

```bash
# 详细输出
make test-verbose

# 静默模式
make test-quiet

# 列出所有测试用例
make test-list

# 自定义Python解释器
make test PY=python3
```

### 直接使用测试运行器

```bash
# 运行所有测试
python3 tests/test_runner.py

# 运行特定类别
python3 tests/test_runner.py --category utils

# 运行特定模块
python3 tests/test_runner.py --module tests.utils.test_cn_text

# 运行特定测试方法
python3 tests/test_runner.py --module tests.utils.test_cn_text.TestChineseTextProcessing.test_to_halfwidth

# 禁用彩色输出
python3 tests/test_runner.py --no-color

# 调整详细程度
python3 tests/test_runner.py --verbosity 0  # 静默
python3 tests/test_runner.py --verbosity 1  # 正常
python3 tests/test_runner.py --verbosity 2  # 详细
```

## 🎯 测试设计原则

### 1. **全面覆盖**
- 核心功能逻辑测试
- 边界情况处理
- 错误恢复机制
- 参数验证

### 2. **独立性**
- 每个测试用例独立运行
- 不依赖外部状态
- 使用Mock隔离依赖
- 临时文件自动清理

### 3. **可读性**
- 描述性测试名称
- 清晰的测试结构
- 详细的注释说明
- 子测试组织

### 4. **中文支持**
- 专门的中文文本测试
- Unicode处理验证
- 编码兼容性测试
- 中文业务逻辑测试

## 🔧 Mock策略

### 外部依赖Mock
- **pandas**: 数据框操作Mock
- **sentence_transformers**: 模型加载Mock
- **torch**: 深度学习框架Mock
- **文件系统**: 临时目录和文件

### Mock技术
- `unittest.mock.patch`: 函数/方法Mock
- `unittest.mock.MagicMock`: 对象Mock
- `tempfile`: 临时文件和目录
- `StringIO`: 内存文件对象

## 📊 测试输出

### 彩色输出示例
```
🧪 QA Clean 项目单元测试
==================================================
📊 发现 25 个测试用例

✓ test_to_halfwidth - 测试全角转半角功能
✓ test_normalize_zh - 测试中文文本归一化功能
✓ test_filter_reason_whitelist - 测试白名单过滤逻辑
- test_large_file_processing - SKIPPED: 需要大文件测试数据

==================================================
📈 测试总结
--------------------------------------------------
总测试数: 25
✓ 成功: 23
- 跳过: 2
⏱️  耗时: 0.12秒
📊 成功率: 92.0%
```

## 🚧 依赖要求

### 最小依赖
测试可以在最小Python环境中运行，核心逻辑测试不依赖外部库：
- Python 3.11+ 标准库
- unittest (内置)
- tempfile (内置)
- os, sys (内置)

### 完整测试依赖
完整功能测试需要项目依赖：
- pandas: 数据框测试
- numpy: 数组操作测试
- yaml: 配置文件测试
- 其他项目依赖

## 🎨 自定义测试

### 添加新测试模块

1. 在适当目录创建 `test_*.py` 文件
2. 继承 `unittest.TestCase`
3. 编写测试方法 (以 `test_` 开头)
4. 添加到Makefile (可选)

示例：
```python
import unittest

class TestNewFeature(unittest.TestCase):
    def test_new_function(self):
        """测试新功能"""
        result = new_function("input")
        self.assertEqual(result, "expected")

if __name__ == '__main__':
    unittest.main()
```

### 自定义测试运行器

可以扩展 `tests/test_runner.py` 添加新功能：
- 测试报告生成
- 性能测试支持
- 集成测试分类
- 自定义输出格式

## 📝 注意事项

1. **环境隔离**: 测试使用临时目录，不影响实际数据
2. **跳过机制**: 缺少依赖时自动跳过相关测试
3. **中文编码**: 确保终端支持UTF-8显示中文测试结果
4. **Python版本**: 针对Python 3.11+设计和测试

## 🔍 故障排除

### 常见问题

1. **Import错误**: 检查Python路径和依赖安装
2. **编码问题**: 确保终端UTF-8编码
3. **权限错误**: 检查测试目录写入权限
4. **依赖缺失**: 使用conda安装完整依赖

### 调试技巧

1. 使用 `--verbosity 2` 查看详细输出
2. 运行单个测试方法定位问题
3. 检查临时文件内容调试
4. 使用IDE断点调试复杂逻辑

---

单元测试确保了QA Clean项目的代码质量和功能正确性，为持续开发和重构提供了可靠保障。
