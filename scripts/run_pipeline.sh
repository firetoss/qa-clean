#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-"src/configs/config.yaml"}

echo "[run_pipeline] use config: ${CFG}"
python -m src.stages.stage1_filter --config "${CFG}"
python -m src.stages.stage2_recall --config "${CFG}"
python -m src.stages.stage3_rerank --config "${CFG}"
python -m src.stages.stage4_cluster --config "${CFG}"
python -m src.stages.stage5_answer_govern --config "${CFG}"

echo "[run_pipeline] done"
