#!/bin/bash
# =============================================================================
# STANDALONE small-matrix BERT4Rec evaluation (eval-only, existing checkpoints).
# Outputs: baseline_results_<variant>/bert4rec_results_small.txt
# Usage: bash eval_bert4rec_small.sh [GPU_ID]
# =============================================================================
cd "$(dirname "$0")"

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

ITEM_NUM=10728
MAXLEN=50
N_CAT=31

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

for VAR in "${VARIANTS[@]}"; do
    DATA_DIR="./KuaiRec_small_eval/${VAR}"
    OUT_DIR="./baseline_results_${VAR}"
    mkdir -p "${OUT_DIR}"

    if [ ! -f "${DATA_DIR}/test-v0.txt" ]; then echo "SKIP: missing ${DATA_DIR}/test-v0.txt"; continue; fi

    BERT_DIR="./save_bert4rec_${VAR}"
    if [ ! -f "${BERT_DIR}/bert4rec_best.pth" ]; then
        echo "[BERT4Rec ${VAR}] no checkpoint — skip"
        continue
    fi

    echo "=== BERT4Rec small-matrix: ${VAR} ==="
    python3 bert4rec_pytorch.py \
        --eval_only --ckpt_dir "${BERT_DIR}" \
        --test_file "${DATA_DIR}/test-v0.txt" \
        --item_num ${ITEM_NUM} --batch_size 256 --maxlen ${MAXLEN} \
        --cat "./KuaiRec_variants/${VAR}/kuairec_cate.txt" \
        --n_cat ${N_CAT} \
        --vec "./KuaiRec_variants/kuairec_vec.npy" \
        --output "${OUT_DIR}/bert4rec_results_small.txt" 2>&1 | tee "eval_small_bert4rec_${VAR}.log"
done

echo "BERT4REC SMALL-MATRIX EVALS DONE"
