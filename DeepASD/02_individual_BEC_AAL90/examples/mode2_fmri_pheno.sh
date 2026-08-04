#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${PROJECT_ROOT}/results/02_fmri_pheno_sparse_directional"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_experiment.py" \
  --experiment_mode 2 \
  --roi_root "${REPO_ROOT}/dataset/ABIDE-I/cpac/filt_noglobal" \
  --phenotypic_csv "${REPO_ROOT}/dataset/ABIDE-I/Phenotypic_V1_0b_preprocessed1.csv" \
  --result_dir "${RESULT_DIR}" \
  --num_rois 90 \
  --folds 5 \
  --epochs 100 \
  --patience 4 \
  --window_length 78 \
  --eval_windows 3 \
  --batch_size 32 \
  --max_incoming_edges 20 \
  --lambda_sparse 0.05 \
  --lambda_directional 0.1 \
  --edge_presence_threshold 0.001 \
  --seed 42

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/classify_bec.py" \
  --result_dir "${RESULT_DIR}" \
  --folds 5 \
  --seed 42 \
  --output_csv "${RESULT_DIR}/classification_metrics.csv"
