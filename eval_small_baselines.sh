#!/bin/bash
# =============================================================================
# Small-matrix (canonical KuaiRec) evaluation of the GRU4Rec + SASRec baselines
# (eval-only, existing checkpoints). Mirrors eval_all.sh's baseline invocation
# with the test file swapped to the small-matrix leave-last-out split.
# BERT4Rec: bash eval_bert4rec_small.sh
#
# Outputs (original *_results.txt files are not touched):
#   baseline_results_<variant>/gru4rec_results_small.txt
#   baseline_results_<variant>/sasrec_results_small.txt
#
# Both report recall/MRR/NDCG + ILD/CS/CC via the shared evaluator.
#
# Usage: bash eval_small_baselines.sh [GPU_ID]
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

    echo "=============================================="
    echo "SMALL-matrix baselines: ${VAR}"
    echo "=============================================="

    # GRU4Rec (per-variant checkpoint)
    GRU_DIR="./save_gru4rec_${VAR}"
    if [ -f "${GRU_DIR}/gru4rec_best.pth" ]; then
        echo "[GRU4Rec] Evaluating..."
        python3 gru4rec_pytorch.py \
            --eval_only --ckpt_dir "${GRU_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --batch_size 256 \
            --maxlen ${MAXLEN} \
            --cat "./KuaiRec_variants/${VAR}/kuairec_cate.txt" \
            --n_cat ${N_CAT} \
            --vec "./KuaiRec_variants/kuairec_vec.npy" \
            --output "${OUT_DIR}/gru4rec_results_small.txt" 2>&1 | tee "eval_small_gru4rec_${VAR}.log"
    else
        echo "[GRU4Rec] No checkpoint for ${VAR} — skip"
    fi

    # SASRec (per-variant checkpoint, same fallback as eval_all.sh)
    SAS_DIR="./save_sasrec_${VAR}"
    if [ -f "${SAS_DIR}/sasrec_best.pth" ]; then
        echo "[SASRec] Evaluating..."
        python3 sasrec_pytorch.py \
            --eval_only --ckpt_dir "${SAS_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --batch_size 256 \
            --maxlen ${MAXLEN} \
            --cat "./KuaiRec_variants/${VAR}/kuairec_cate.txt" \
            --n_cat ${N_CAT} \
            --vec "./KuaiRec_variants/kuairec_vec.npy" \
            --output "${OUT_DIR}/sasrec_results_small.txt" 2>&1 | tee "eval_small_sasrec_${VAR}.log"
    elif [ -f "./sasrec_best.pth" ]; then
        echo "[SASRec] Evaluating (shared checkpoint sasrec_best.pth)..."
        python3 sasrec_pytorch.py \
            --eval_only --ckpt_path sasrec_best.pth --ckpt_dir "." \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --batch_size 256 \
            --maxlen ${MAXLEN} \
            --cat "./KuaiRec_variants/${VAR}/kuairec_cate.txt" \
            --n_cat ${N_CAT} \
            --vec "./KuaiRec_variants/kuairec_vec.npy" \
            --output "${OUT_DIR}/sasrec_results_small.txt" 2>&1 | tee "eval_small_sasrec_${VAR}.log"
    else
        echo "[SASRec] No checkpoint — skip"
    fi

    # NOTE: BERT4Rec small-matrix eval is standalone: eval_bert4rec_small.sh

    echo "=== ${VAR} baselines done ==="
    echo ""
done

echo "ALL SMALL-MATRIX BASELINE EVALS DONE (GRU4Rec + SASRec; BERT4Rec: eval_bert4rec_small.sh)"
