# Makefile for Chinese QA Clean Pipeline (FAISS-only)

PY ?= python
CFG ?= src/configs/config.yaml
INPUT ?= 

.PHONY: help init pipeline run stage1 stage2 stage3 stage4 stage5 clean check env-cpu env-gpu

help: ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

init: ## 初始化目录结构（data/raw 与 outputs/figs）
	@mkdir -p data/raw outputs/figs
	@echo "[init] 确保目录已就绪：data/raw, outputs/figs"

pipeline: ## 通过脚本顺序运行五个阶段
	@if [ -n "$(INPUT)" ]; then \
		bash scripts/run_pipeline.sh $(CFG) $(INPUT); \
	else \
		bash scripts/run_pipeline.sh $(CFG); \
	fi

run: ## 通过模块一键运行（与 pipeline 等价）
	@if [ -n "$(INPUT)" ]; then \
		$(PY) src/run_all.py --config $(CFG) --input $(INPUT); \
	else \
		$(PY) src/run_all.py --config $(CFG); \
	fi

stage1: ## 运行阶段1：字符级预处理与过滤
	@if [ -n "$(INPUT)" ]; then \
		$(PY) -m src.stages.stage1_filter --config $(CFG) --input $(INPUT); \
	else \
		$(PY) -m src.stages.stage1_filter --config $(CFG); \
	fi

stage2: ## 运行阶段2：三路嵌入+FAISS召回+字符n-gram补召
	@if [ -n "$(INPUT)" ]; then \
		$(PY) -m src.stages.stage2_recall --config $(CFG) --input $(INPUT); \
	else \
		$(PY) -m src.stages.stage2_recall --config $(CFG); \
	fi

stage3: ## 运行阶段3：多交叉编码器融合精排
	@if [ -n "$(INPUT)" ]; then \
		$(PY) -m src.stages.stage3_rerank --config $(CFG) --input $(INPUT); \
	else \
		$(PY) -m src.stages.stage3_rerank --config $(CFG); \
	fi

stage4: ## 运行阶段4：图聚类 + 中心约束 + 二次聚合
	@if [ -n "$(INPUT)" ]; then \
		$(PY) -m src.stages.stage4_cluster --config $(CFG) --input $(INPUT); \
	else \
		$(PY) -m src.stages.stage4_cluster --config $(CFG); \
	fi

stage5: ## 运行阶段5：答案治理与融合
	@if [ -n "$(INPUT)" ]; then \
		$(PY) -m src.stages.stage5_answer_govern --config $(CFG) --input $(INPUT); \
	else \
		$(PY) -m src.stages.stage5_answer_govern --config $(CFG); \
	fi

clean: ## 清理 outputs 产物
	@rm -rf outputs/*
	@mkdir -p outputs/figs
	@echo "[clean] 已清理 outputs/*"

check: ## 依赖自检（导入核心库）
	@echo "[check] 检查依赖..."
	@$(PY) -c "import numpy, pandas, pyarrow, sklearn, tqdm, yaml, regex, sentence_transformers, torch; print('[check] 依赖检查通过')"

test: ## 运行所有单元测试
	@echo "[test] 运行单元测试..."
	@$(PY) tests/test_runner.py

test-utils: ## 运行工具模块测试
	@echo "[test-utils] 运行工具模块测试..."
	@$(PY) tests/test_runner.py --category utils

test-stages: ## 运行阶段模块测试
	@echo "[test-stages] 运行阶段模块测试..."
	@$(PY) tests/test_runner.py --category stages

test-verbose: ## 运行详细测试（包含跳过的测试信息）
	@echo "[test-verbose] 运行详细测试..."
	@$(PY) tests/test_runner.py --verbosity 2

test-quiet: ## 运行静默测试
	@echo "[test-quiet] 运行静默测试..."
	@$(PY) tests/test_runner.py --verbosity 0

test-list: ## 列出所有测试用例
	@echo "[test-list] 列出测试用例..."
	@$(PY) tests/test_runner.py --list

test-cn-text: ## 运行中文文本处理测试
	@echo "[test-cn-text] 运行中文文本处理测试..."
	@$(PY) tests/test_runner.py --module tests.utils.test_cn_text

test-io: ## 运行IO工具测试
	@echo "[test-io] 运行IO工具测试..."
	@$(PY) tests/test_runner.py --module tests.utils.test_io_utils

test-config: ## 运行配置管理测试
	@echo "[test-config] 运行配置管理测试..."
	@$(PY) tests/test_runner.py --module tests.utils.test_config

test-sim: ## 运行文本相似度测试
	@echo "[test-sim] 运行文本相似度测试..."
	@$(PY) tests/test_runner.py --module tests.utils.test_text_sim

test-stage1: ## 运行Stage1过滤测试
	@echo "[test-stage1] 运行Stage1过滤测试..."
	@$(PY) tests/test_runner.py --module tests.stages.test_stage1_filter

env-cpu: ## 提示CPU环境安装命令（使用conda-forge）
	@echo "conda create -n qa-clean-pipe python=3.11 -y && conda activate qa-clean-pipe"; \
	echo "conda install -c conda-forge faiss-cpu numpy pandas pyarrow openpyxl scikit-learn tqdm pyyaml regex -y"; \
	echo "pip install sentence-transformers torch matplotlib rapidfuzz"

env-gpu: ## 提示GPU环境安装命令（使用conda-forge）
	@echo "conda create -n qa-clean-pipe python=3.11 -y && conda activate qa-clean-pipe"; \
	echo "conda install -c conda-forge faiss-gpu numpy pandas pyarrow openpyxl scikit-learn tqdm pyyaml regex -y"; \
	echo "pip install sentence-transformers torch matplotlib rapidfuzz"
