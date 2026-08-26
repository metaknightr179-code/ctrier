#!/bin/bash
# =============================================================================
# Run all 3 baselines (GRU4Rec, SASRec, BERT4Rec) on all 4 Kuairec variants.
# Computes the same metrics as TRIER (recall, MRR, NDCG, ILD, CS, CC).
# Usage: bash run_all_baselines.sh
# =============================================================================

set -e

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

GPU=0
ITEM_NUM=10728
N_CAT=31
EPOCHS=500
BATCH_SIZE=256
LR=1e-3
MAXLEN=50

FAILED=""

for VAR in "${VARIANTS[@]}"; do
    DATA_DIR="./KuaiRec_variants/${VAR}"
    OUTPUT_DIR="./baseline_results_${VAR}"
    mkdir -p "${OUTPUT_DIR}"

    echo "############################################################"
    echo "# Baselines for: ${VAR}"
    echo "############################################################"

    # ===================== GRU4Rec =====================
    echo "=============================================="
    echo "  GRU4Rec — ${VAR}"
    echo "=============================================="
    set +e
    CUDA_VISIBLE_DEVICES=${GPU} python3 gru4rec_pytorch.py \
        --train_file "${DATA_DIR}/train-v0.txt" \
        --test_file "${DATA_DIR}/test-v0.txt" \
        --item_num ${ITEM_NUM} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --maxlen ${MAXLEN} \
        --cat "${DATA_DIR}/kuairec_cate.txt" \
        --n_cat ${N_CAT} \
        --vec "./KuaiRec_variants/kuairec_vec.npy" \
        --ckpt_dir "./save_gru4rec_${VAR}" \
        --output "${OUTPUT_DIR}/gru4rec_results.txt" 2>&1 | tee "${OUTPUT_DIR}/gru4rec.log"
    if [ $? -ne 0 ]; then
        echo "ERROR: GRU4Rec failed for ${VAR}"
        FAILED="${FAILED} gru4rec/${VAR}"
    fi
    set -e

    # ===================== SASRec =====================
    echo "=============================================="
    echo "  SASRec — ${VAR}"
    echo "=============================================="
    set +e
    CUDA_VISIBLE_DEVICES=${GPU} python3 sasrec_pytorch.py \
        --train_file "${DATA_DIR}/train-v0.txt" \
        --test_file "${DATA_DIR}/test-v0.txt" \
        --item_num ${ITEM_NUM} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --maxlen ${MAXLEN} \
        --ckpt_dir "./save_sasrec_${VAR}" \
        --output "${OUTPUT_DIR}/sasrec_results.txt" 2>&1 | tee "${OUTPUT_DIR}/sasrec.log"
    if [ $? -ne 0 ]; then
        echo "ERROR: SASRec failed for ${VAR}"
        FAILED="${FAILED} sasrec/${VAR}"
    fi
    set -e

    # ===================== BERT4Rec =====================
    echo "=============================================="
    echo "  BERT4Rec — ${VAR}"
    echo "=============================================="
    set +e
    CUDA_VISIBLE_DEVICES=${GPU} python3 bert4rec_pytorch.py \
        --train_file "${DATA_DIR}/train-v0.txt" \
        --test_file "${DATA_DIR}/test-v0.txt" \
        --item_num ${ITEM_NUM} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --maxlen ${MAXLEN} \
        --ckpt_dir "./save_bert4rec_${VAR}" \
        --output "${OUTPUT_DIR}/bert4rec_results.txt" 2>&1 | tee "${OUTPUT_DIR}/bert4rec.log"
    if [ $? -ne 0 ]; then
        echo "ERROR: BERT4Rec failed for ${VAR}"
        FAILED="${FAILED} bert4rec/${VAR}"
    fi
    set -e

    echo ""
done

echo "############################################################"
if [ -n "$FAILED" ]; then
    echo "BASELINES FINISHED WITH FAILURES:"
    echo "${FAILED}"
    echo "Check logs in baseline_results_<variant>/"
else
    echo "ALL BASELINES COMPLETE!"
fi
echo "############################################################"
echo ""
echo "Results:"
for VAR in "${VARIANTS[@]}"; do
    echo "  ${VAR}:"
    echo "    GRU4Rec:  ./baseline_results_${VAR}/gru4rec_results.txt"
    echo "    SASRec:   ./baseline_results_${VAR}/sasrec_results.txt"
    echo "    BERT4Rec: ./baseline_results_${VAR}/bert4rec_results.txt"
done
