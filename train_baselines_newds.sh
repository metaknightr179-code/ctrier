#!/bin/bash
# Train SASRec + GRU4Rec baselines on the new datasets (ML1M, KuaiRand1K, MicroLens).
#
# Usage:  bash train_baselines_newds.sh <GPU_ID> <DATASET>
#         DATASET in {ML1M, KuaiRand1K, MicroLens}
#
# Protocol mirrors the KuaiRec baselines: 500 epochs, batch 256, lr 1e-3.
# Results -> baseline_results_<DS>/, checkpoints -> save_sasrec_<DS>/ save_gru4rec_<DS>/

set -u
GPU=${1:?Usage: train_baselines_newds.sh <GPU_ID> <DATASET>}
DS=${2:?DATASET must be ML1M, KuaiRand1K or MicroLens}

case "$DS" in
  ML1M)
    N=3126; NCAT=18; DIR=./ML1M; CATE=ml1m_cate.txt; VEC=ml1m_vec.npy ;;
  KuaiRand1K)
    N=20001; NCAT=44; DIR=./KuaiRand1K; CATE=kuairand_cate.txt; VEC=kuairand_vec.npy ;;
  MicroLens)
    N=26923; NCAT=57; DIR=./MicroLens; CATE=microlens_cate.txt; VEC=microlens_vec.npy ;;
  *) echo "Unknown DATASET: $DS"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES=${GPU}

# Guard: data must be present
for f in "${DIR}/train-v0.txt" "${DIR}/test-v0.txt" "${DIR}/${CATE}" "${DIR}/${VEC}"; do
  [ -f "$f" ] || { echo "ERROR: missing $f - aborting"; exit 1; }
done

OUT_DIR="./baseline_results_${DS}"
mkdir -p "${OUT_DIR}"

# Batch 256 fits all converted datasets after the popularity cut
BATCH=256

echo "=============================================="
echo "Baselines - ${DS} (item_num=${N}, n_cat=${NCAT}, batch=${BATCH})"
echo "=============================================="

# ---------------- GRU4Rec ----------------
GRU_DIR="./save_gru4rec_${DS}"
if [ -f "${GRU_DIR}/gru4rec_best.pth" ]; then
  echo "[GRU4Rec] checkpoint exists - skipping training"
else
  echo "[GRU4Rec] training..."
  python3 gru4rec_pytorch.py \
    --train_file "${DIR}/train-v0.txt" \
    --valid_file "${DIR}/valid-v0.txt" \
    --test_file "${DIR}/test-v0.txt" \
    --item_num ${N} \
    --epochs 500 --batch_size ${BATCH} --lr 1e-3 --maxlen 50 \
    --patience 100 \
    --cat "${DIR}/${CATE}" --n_cat ${NCAT} --vec "${DIR}/${VEC}" \
    --ckpt_dir "${GRU_DIR}" \
    --output "${OUT_DIR}/gru4rec_results.txt" 2>&1 | tee "train_gru4rec_${DS}.log"
fi

# ---------------- SASRec ----------------
SAS_DIR="./save_sasrec_${DS}"
if [ -f "${SAS_DIR}/sasrec_best.pth" ]; then
  echo "[SASRec] checkpoint exists - skipping training"
else
  echo "[SASRec] training..."
  python3 sasrec_pytorch.py \
    --train_file "${DIR}/train-v0.txt" \
    --valid_file "${DIR}/valid-v0.txt" \
    --test_file "${DIR}/test-v0.txt" \
    --item_num ${N} \
    --epochs 500 --batch_size ${BATCH} --lr 1e-3 --maxlen 50 \
    --patience 100 \
    --cat "${DIR}/${CATE}" --n_cat ${NCAT} --vec "${DIR}/${VEC}" \
    --ckpt_dir "${SAS_DIR}" \
    --output "${OUT_DIR}/sasrec_results.txt" 2>&1 | tee "train_sasrec_${DS}.log"
fi

echo "BASELINES [${DS}] COMPLETE - results in ${OUT_DIR}/"
