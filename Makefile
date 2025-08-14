# Makefile for Chinese QA Clean Pipeline (FAISS-only)

PY ?= python
CFG ?= src/configs/config.yaml

.PHONY: help init pipeline run stage1 stage2 stage3 stage4 stage5 clean check env-cpu env-gpu

help: ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

init: ## 初始化目录结构（data/raw 与 outputs/figs）
	@mkdir -p data/raw outputs/figs
	@echo "[init] 确保目录已就绪：data/raw, outputs/figs"

pipeline: ## 通过脚本顺序运行五个阶段
	@bash scripts/run_pipeline.sh $(CFG)

run: ## 通过模块一键运行（与 pipeline 等价）
	@$(PY) src/run_all.py $(CFG)

stage1: ## 运行阶段1：字符级预处理与过滤
	@$(PY) -m src.stages.stage1_filter --config $(CFG)

stage2: ## 运行阶段2：三路嵌入+FAISS召回+字符n-gram补召
	@$(PY) -m src.stages.stage2_recall --config $(CFG)

stage3: ## 运行阶段3：多交叉编码器融合精排
	@$(PY) -m src.stages.stage3_rerank --config $(CFG)

stage4: ## 运行阶段4：图聚类 + 中心约束 + 二次聚合
	@$(PY) -m src.stages.stage4_cluster --config $(CFG)

stage5: ## 运行阶段5：答案治理与融合
	@$(PY) -m src.stages.stage5_answer_govern --config $(CFG)

clean: ## 清理 outputs 产物
	@rm -rf outputs/*
	@mkdir -p outputs/figs
	@echo "[clean] 已清理 outputs/*"

check: ## 依赖自检（导入核心库）
	@$(PY) - <<'PY'
import importlib, sys
mods = [
    'numpy','pandas','pyarrow','sklearn','tqdm','yaml','regex',
    'sentence_transformers'
]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, str(e)))
if failed:
    print('[check] 缺少依赖:')
    for m, e in failed:
        print('  -', m, '->', e)
    sys.exit(1)
else:
    print('[check] 依赖检查通过')
PY

env-cpu: ## 提示CPU环境安装命令（使用conda-forge）
	@echo "conda create -n qa-clean-pipe python=3.11 -y && conda activate qa-clean-pipe"; \
	echo "conda install -c conda-forge faiss-cpu numpy pandas pyarrow scikit-learn tqdm pyyaml regex -y"; \
	echo "pip install sentence-transformers matplotlib rapidfuzz"

env-gpu: ## 提示GPU环境安装命令（使用conda-forge）
	@echo "conda create -n qa-clean-pipe python=3.11 -y && conda activate qa-clean-pipe"; \
	echo "conda install -c conda-forge faiss-gpu numpy pandas pyarrow scikit-learn tqdm pyyaml regex -y"; \
	echo "pip install sentence-transformers matplotlib rapidfuzz"
