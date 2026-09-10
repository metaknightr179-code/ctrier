#!/bin/bash
# =============================================================================
# Big-matrix evaluation of baselines (GRU4Rec, SASRec) on KuaiRec.
# BERT4Rec: bash eval_bert4rec_big.sh   (TRIER PT: eval_dense_kuairec.sh)
#
# Usage:
#   bash eval_all.sh
#   nohup bash eval_all.sh > eval_all.log 2>&1 &
# =============================================================================

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
MAXLEN=50

echo "############################################################"
echo "# Baselines (big matrix, eval-only)"
echo "############################################################"
echo ""

for VAR in "${VARIANTS[@]}"; do
    DATA_DIR="./KuaiRec_variants/${VAR}"
    OUTPUT_DIR="./baseline_results_${VAR}"
    mkdir -p "${OUTPUT_DIR}"

    echo "=============================================="
    echo "Baselines — ${VAR}"
    echo "=============================================="

    # GRU4Rec
    GRU_DIR="./save_gru4rec_${VAR}"
    if [ -f "${GRU_DIR}/gru4rec_best.pth" ]; then
        echo "[GRU4Rec] Evaluating..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 gru4rec_pytorch.py \
            --eval_only --ckpt_dir "${GRU_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --batch_size 256 \
            --maxlen ${MAXLEN} \
            --cat "${DATA_DIR}/kuairec_cate.txt" \
            --n_cat ${N_CAT} \
            --vec "./KuaiRec_variants/kuairec_vec.npy" \
            --output "${OUTPUT_DIR}/gru4rec_results.txt" 2>&1 | tee "eval_gru4rec_${VAR}.log"
    else
        echo "[GRU4Rec] No checkpoint — skip"
    fi

    # SASRec
    SAS_DIR="./save_sasrec_${VAR}"
    if [ -f "${SAS_DIR}/sasrec_best.pth" ]; then
        echo "[SASRec] Evaluating..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 sasrec_pytorch.py \
            --eval_only --ckpt_dir "${SAS_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUTPUT_DIR}/sasrec_results.txt" 2>&1 | tee "eval_sasrec_${VAR}.log"
    elif [ -f "./sasrec_best.pth" ]; then
        echo "[SASRec] Using shared checkpoint..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 sasrec_pytorch.py \
            --eval_only --ckpt_path "sasrec_best.pth" --ckpt_dir "." \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUTPUT_DIR}/sasrec_results.txt" 2>&1 | tee "eval_sasrec_${VAR}.log"
    else
        echo "[SASRec] No checkpoint — skip"
    fi

    # NOTE: BERT4Rec is evaluated standalone via eval_bert4rec_big.sh

    echo "=== ${VAR} baselines complete ==="
    echo ""
done

echo "############################################################"
echo "BASELINE EVALUATION COMPLETE!"
echo "############################################################"
echo ""
echo "Results:"
for VAR in "${VARIANTS[@]}"; do
    echo "  Baselines ${VAR}: ./baseline_results_${VAR}/"
done