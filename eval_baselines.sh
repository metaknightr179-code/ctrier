#!/bin/bash
# =============================================================================
# Re-evaluate all baselines with fixed NDCG formula.
# - GRU4Rec: eval-only using per-variant checkpoints
# - SASRec/BERT4Rec: eval-only if checkpoints exist, otherwise retrain + eval
# Usage: bash eval_baselines.sh
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
EPOCHS=500
BATCH_SIZE=256
LR=1e-3

echo "=============================================="
echo "Re-evaluating baselines with fixed NDCG"
echo "=============================================="

for VAR in "${VARIANTS[@]}"; do
    DATA_DIR="./KuaiRec_variants/${VAR}"
    OUTPUT_DIR="./baseline_results_${VAR}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "=============================================="
    echo "Variant: ${VAR}"
    echo "=============================================="

    # --- GRU4Rec (eval-only, uses save_gru4rec_<VAR>/) ---
    GRU_DIR="./save_gru4rec_${VAR}"
    GRU_CKPT="${GRU_DIR}/gru4rec_best.pth"
    if [ -f "${GRU_CKPT}" ]; then
        echo "[GRU4Rec] Re-evaluating with fixed NDCG..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 gru4rec_pytorch.py \
            --eval_only --ckpt_path "gru4rec_best.pth" --ckpt_dir "${GRU_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --batch_size ${BATCH_SIZE} \
            --maxlen ${MAXLEN} \
            --cat "${DATA_DIR}/kuairec_cate.txt" \
            --n_cat ${N_CAT} \
            --vec "./KuaiRec_variants/kuairec_vec.npy" \
            --output "${OUTPUT_DIR}/gru4rec_results.txt" 2>&1 | tee "eval_gru4rec_${VAR}.log"
    else
        echo "[GRU4Rec] Checkpoint not found: ${GRU_CKPT} — skip (run training first)"
    fi

    # --- SASRec (eval-only if checkpoint exists, else retrain) ---
    SAS_DIR="./save_sasrec_${VAR}"
    SAS_CKPT="${SAS_DIR}/sasrec_best.pth"
    if [ -f "${SAS_CKPT}" ]; then
        echo "[SASRec] Re-evaluating with fixed NDCG..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 sasrec_pytorch.py \
            --eval_only --ckpt_dir "${SAS_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUTPUT_DIR}/sasrec_results.txt" 2>&1 | tee "eval_sasrec_${VAR}.log"
    else
        echo "[SASRec] No per-variant checkpoint — training + evaluating..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 sasrec_pytorch.py \
            --train_file "${DATA_DIR}/train-v0.txt" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --maxlen ${MAXLEN} \
            --ckpt_dir "${SAS_DIR}" \
            --output "${OUTPUT_DIR}/sasrec_results.txt" 2>&1 | tee "eval_sasrec_${VAR}.log"
    fi

    # --- BERT4Rec (eval-only if checkpoint exists, else retrain) ---
    BERT_DIR="./save_bert4rec_${VAR}"
    BERT_CKPT="${BERT_DIR}/bert4rec_best.pth"
    if [ -f "${BERT_CKPT}" ]; then
        echo "[BERT4Rec] Re-evaluating with fixed NDCG..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 bert4rec_pytorch.py \
            --eval_only --ckpt_dir "${BERT_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUTPUT_DIR}/bert4rec_results.txt" 2>&1 | tee "eval_bert4rec_${VAR}.log"
    else
        echo "[BERT4Rec] No per-variant checkpoint — training + evaluating..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 bert4rec_pytorch.py \
            --train_file "${DATA_DIR}/train-v0.txt" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --maxlen ${MAXLEN} \
            --ckpt_dir "${BERT_DIR}" \
            --output "${OUTPUT_DIR}/bert4rec_results.txt" 2>&1 | tee "eval_bert4rec_${VAR}.log"
    fi

    echo "=== Variant ${VAR} complete ==="
done

echo ""
echo "=============================================="
echo "ALL BASELINE RE-EVALUATIONS COMPLETE!"
echo "=============================================="
echo ""
echo "Updated results:"
for VAR in "${VARIANTS[@]}"; do
    echo "  ${VAR}:"
    echo "    GRU4Rec:  ./baseline_results_${VAR}/gru4rec_results.txt"
    echo "    SASRec:   ./baseline_results_${VAR}/sasrec_results.txt"
    echo "    BERT4Rec: ./baseline_results_${VAR}/bert4rec_results.txt"
done