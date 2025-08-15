#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-"src/configs/config.yaml"}
INPUT=${2:-""}

echo "[run_pipeline] use config: ${CFG}"
if [ -n "${INPUT}" ]; then
    echo "[run_pipeline] use input file: ${INPUT}"
    python -m src.stages.stage1_filter --config "${CFG}" --input "${INPUT}"
    python -m src.stages.stage2_recall --config "${CFG}" --input "${INPUT}"
    python -m src.stages.stage3_rerank --config "${CFG}" --input "${INPUT}"
    python -m src.stages.stage4_cluster --config "${CFG}" --input "${INPUT}"
    python -m src.stages.stage5_answer_govern --config "${CFG}" --input "${INPUT}"
else
    python -m src.stages.stage1_filter --config "${CFG}"
    python -m src.stages.stage2_recall --config "${CFG}"
    python -m src.stages.stage3_rerank --config "${CFG}"
    python -m src.stages.stage4_cluster --config "${CFG}"
    python -m src.stages.stage5_answer_govern --config "${CFG}"
fi

echo "[run_pipeline] done"
