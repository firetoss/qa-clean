#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-"src/configs/config.yaml"}

echo "[run_pipeline] use config: ${CFG}"
python src/stages/stage1_filter.py --config "${CFG}"
python src/stages/stage2_recall.py --config "${CFG}"
python src/stages/stage3_rerank.py --config "${CFG}"
python src/stages/stage4_cluster.py --config "${CFG}"
python src/stages/stage5_answer_govern.py --config "${CFG}"

echo "[run_pipeline] done"
