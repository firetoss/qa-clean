# Makefile for QA Clean Pipeline - Simplified

.PHONY: help init run clean check test

help: ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

init: ## 初始化目录结构
	@mkdir -p data/raw outputs/figs outputs/logs
	@echo "[init] 目录初始化完成"

run: ## 运行完整流水线 (使用: make run INPUT=data/raw/qa.xlsx)
	@if [ -z "$(INPUT)" ]; then \
		echo "错误: 请指定输入文件，例如: make run INPUT=data/raw/qa.xlsx"; \
		exit 1; \
	fi
	@python run.py -i $(INPUT)

clean: ## 清理输出目录
	@rm -rf outputs/*
	@mkdir -p outputs/figs outputs/logs
	@echo "[clean] 输出目录已清理"

check: ## 检查环境和依赖
	@echo "[check] 检查Python环境和核心依赖..."
	@python -c "import sys; print(f'Python版本: {sys.version}'); import torch, numpy, pandas, sklearn, sentence_transformers, faiss; print('✅ 核心依赖检查通过')"

test: ## 运行测试
	@echo "[test] 运行测试套件..."
	@python tests/test_runner.py