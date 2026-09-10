#!/bin/bash
# =============================================================================
# STANDALONE BERT4Rec training on KuaiRec — intentionally SEPARATE from
# train_baselines_kuairec.sh (SASRec/GRU4Rec), since BERT4Rec is much slower.
#
# Usage:
#   bash train_bert4rec_kuairec.sh <GPU_ID>
#   nohup bash train_bert4rec_kuairec.sh 0 > train_bert4rec_kuairec.log 2>&1 &
#
# Big-matrix results -> baseline_results_<variant>/bert4rec_results.txt
# Small matrix:       bash eval_bert4rec_small.sh
# Checkpoints      -> save_bert4rec_<variant>/bert4rec_best.pth
# Skips variants that already have a checkpoint.
# =============================================================================
set -u
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

ITEM_NUM=10728
N_CAT=31
MAXLEN=50
BATCH=256
EPOCHS=500
VEC="./KuaiRec_variants/kuairec_vec.npy"

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

for VAR in "${VARIANTS[@]}"; do
    DATA_DIR="./KuaiRec_variants/${VAR}"
    OUT_DIR="./baseline_results_${VAR}"
    mkdir -p "${OUT_DIR}"

    for f in "${DATA_DIR}/train-v0.txt" "${DATA_DIR}/valid-v0.txt" "${DATA_DIR}/test-v0.txt" "${DATA_DIR}/kuairec_cate.txt"; do
        [ -f "$f" ] || { echo "ERROR: missing $f - abort"; exit 1; }
    done

    BERT_DIR="./save_bert4rec_${VAR}"
    if [ -f "${BERT_DIR}/bert4rec_best.pth" ]; then
        echo "[BERT4Rec ${VAR}] checkpoint exists - skipping"
        continue
    fi

    echo "############################################################"
    echo "# BERT4Rec standalone — ${VAR}"
    echo "############################################################"
    python3 bert4rec_pytorch.py \
        --train_file "${DATA_DIR}/train-v0.txt" \
        --valid_file "${DATA_DIR}/valid-v0.txt" \
        --test_file "${DATA_DIR}/test-v0.txt" \
        --item_num ${ITEM_NUM} \
        --epochs ${EPOCHS} --batch_size ${BATCH} --lr 1e-3 --maxlen ${MAXLEN} \
        --patience 100 \
        --cat "${DATA_DIR}/kuairec_cate.txt" --n_cat ${N_CAT} --vec "${VEC}" \
        --ckpt_dir "${BERT_DIR}" \
        --output "${OUT_DIR}/bert4rec_results.txt" 2>&1 | tee "train_bert4rec_${VAR}.log"
done

echo "BERT4REC STANDALONE TRAINING COMPLETE"
