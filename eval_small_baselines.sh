#!/bin/bash
# =============================================================================
# Small-matrix (canonical KuaiRec) evaluation of the GRU4Rec + SASRec baselines
# (eval-only, existing checkpoints). Mirrors eval_all.sh's baseline invocation
# with the test file swapped to the small-matrix leave-last-out split.
#
# Outputs (original *_results.txt files are not touched):
#   baseline_results_<variant>/gru4rec_results_small.txt
#   baseline_results_<variant>/sasrec_results_small.txt
#
# SASRec's eval script only reports recall/MRR/NDCG; run
# eval_sasrec_ild_cs_small.py afterwards for its ILD/CS.
#
# Usage: bash eval_small_baselines.sh
# =============================================================================
cd "$(dirname "$0")"

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

    # SASRec (shared root checkpoint, same fallback as eval_all.sh)
    if [ -f "./sasrec_best.pth" ]; then
        echo "[SASRec] Evaluating (shared checkpoint sasrec_best.pth)..."
        python3 sasrec_pytorch.py \
            --eval_only --ckpt_path sasrec_best.pth --ckpt_dir "." \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUT_DIR}/sasrec_results_small.txt" 2>&1 | tee "eval_small_sasrec_${VAR}.log"
    else
        echo "[SASRec] No checkpoint — skip"
    fi

    echo "=== ${VAR} baselines done ==="
    echo ""
done

echo "ALL SMALL-MATRIX BASELINE EVALS DONE"
