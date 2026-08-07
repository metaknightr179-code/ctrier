#!/bin/bash
# Run baseline models on Kuairec first_average variant
# Usage: bash run_baselines.sh

set -e

VAR="kuairec_first_average"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${PROJECT_DIR}/KuaiRec_variants/${VAR}"
OUTPUT_DIR="${PROJECT_DIR}/baseline_results_${VAR}"
ITEM_NUM=10728

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Training GRU4Rec on ${VAR}"
echo "=============================================="

python3 "${PROJECT_DIR}/gru4rec_pytorch.py" \
    --train_file "${DATA_DIR}/train-v0.txt" \
    --test_file "${DATA_DIR}/test-v0.txt" \
    --item_num ${ITEM_NUM} \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001 \
    --output "${OUTPUT_DIR}/gru4rec_results.txt"

echo ""
echo "=============================================="
echo "Training SASRec on ${VAR}"
echo "=============================================="

python3 "${PROJECT_DIR}/sasrec_pytorch.py" \
    --train_file "${DATA_DIR}/train-v0.txt" \
    --test_file "${DATA_DIR}/test-v0.txt" \
    --item_num ${ITEM_NUM} \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001 \
    --maxlen 50 \
    --output "${OUTPUT_DIR}/sasrec_results.txt"

echo ""
echo "=============================================="
echo "All baselines complete!"
echo "=============================================="
echo ""
echo "=== GRU4Rec Results ==="
cat "${OUTPUT_DIR}/gru4rec_results.txt"
echo ""
echo "=== SASRec Results ==="
cat "${OUTPUT_DIR}/sasrec_results.txt"
