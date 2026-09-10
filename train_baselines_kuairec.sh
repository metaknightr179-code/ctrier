#!/bin/bash
# Train SASRec + GRU4Rec + FPMC baselines on KuaiRec (4 variants).
#
# Usage:  bash train_baselines_kuairec.sh <GPU_ID>
#         nohup bash train_baselines_kuairec.sh 0 > train_baselines_kuairec.log 2>&1 &
#
# Protocol: 500 epochs (SASRec/GRU4Rec, early stop patience 100);
#           FPMC converges in a few epochs (patience 30, val R@10).
# Checkpoints -> save_sasrec_<variant>/ save_gru4rec_<variant>/ save_fpmc_<variant>/
# Results     -> baseline_results_<variant>/

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

    # Guard: data must be present
    for f in "${DATA_DIR}/train-v0.txt" "${DATA_DIR}/valid-v0.txt" "${DATA_DIR}/test-v0.txt" "${DATA_DIR}/kuairec_cate.txt"; do
        [ -f "$f" ] || { echo "ERROR: missing $f - abort"; exit 1; }
    done

    echo "############################################################"
    echo "# Baselines — ${VAR} (item_num=${ITEM_NUM}, n_cat=${N_CAT}, batch=${BATCH})"
    echo "############################################################"

    # ---------------- GRU4Rec ----------------
    GRU_DIR="./save_gru4rec_${VAR}"
    if [ -f "${GRU_DIR}/gru4rec_best.pth" ]; then
        echo "[GRU4Rec] checkpoint exists - skipping"
    else
        echo "[GRU4Rec] training..."
        python3 gru4rec_pytorch.py \
            --train_file "${DATA_DIR}/train-v0.txt" \
            --valid_file "${DATA_DIR}/valid-v0.txt" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --epochs ${EPOCHS} --batch_size ${BATCH} --lr 1e-3 --maxlen ${MAXLEN} \
            --patience 100 \
            --cat "${DATA_DIR}/kuairec_cate.txt" --n_cat ${N_CAT} --vec "${VEC}" \
            --ckpt_dir "${GRU_DIR}" \
            --output "${OUT_DIR}/gru4rec_results.txt" 2>&1 | tee "train_gru4rec_${VAR}.log"
    fi

    # ---------------- SASRec ----------------
    SAS_DIR="./save_sasrec_${VAR}"
    if [ -f "${SAS_DIR}/sasrec_best.pth" ]; then
        echo "[SASRec] checkpoint exists - skipping"
    else
        echo "[SASRec] training..."
        python3 sasrec_pytorch.py \
            --train_file "${DATA_DIR}/train-v0.txt" \
            --valid_file "${DATA_DIR}/valid-v0.txt" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --epochs ${EPOCHS} --batch_size ${BATCH} --lr 1e-3 --maxlen ${MAXLEN} \
            --patience 100 \
            --cat "${DATA_DIR}/kuairec_cate.txt" --n_cat ${N_CAT} --vec "${VEC}" \
            --ckpt_dir "${SAS_DIR}" \
            --output "${OUT_DIR}/sasrec_results.txt" 2>&1 | tee "train_sasrec_${VAR}.log"
    fi

    # ---------------- FPMC ----------------
    FPMC_DIR="./save_fpmc_${VAR}"
    if [ -f "${FPMC_DIR}/fpmc_best.pth" ]; then
        echo "[FPMC] checkpoint exists - skipping"
    else
        echo "[FPMC] training..."
        python3 fpmc_pytorch.py \
            --train_file "${DATA_DIR}/train-v0.txt" \
            --valid_file "${DATA_DIR}/valid-v0.txt" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --epochs 200 --batch_size 1024 --lr 1e-2 --dim 64 --n_neg 10 \
            --maxlen ${MAXLEN} --patience 30 \
            --cat "${DATA_DIR}/kuairec_cate.txt" --n_cat ${N_CAT} --vec "${VEC}" \
            --ckpt_dir "${FPMC_DIR}" \
            --output "${OUT_DIR}/fpmc_results.txt" 2>&1 | tee "train_fpmc_${VAR}.log"
    fi

    echo "=== ${VAR} baselines complete ==="
done

echo "ALL KUAIREC BASELINES COMPLETE"
