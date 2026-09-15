#!/bin/bash
# =============================================================================
# Single-config per-dataset retrain — BEST CONFIG from the KuaiRec γ sweep.
#
# γ sweep on kuairec_first_average: NDCG@20 peak at γ_o=0 (no L_order),
# λ=0.01, dense CE. So this script retrains exactly that config (γ=0 = no
# -soft_order_loss flag) on every dataset/variant, one at a time, on a
# single GPU. RT checkpoint is a prerequisite and is retrained if missing.
#
# Pipeline per dataset:
#   Stage 1: RT (save_rt_fix_<DS>) — skipped if DONE exists.
#   Stage 2: PT dense -div -lamb 0.01 (no L_order) — PACER = type family.
#
# Supported DATASET values:
#   ML1M, KuaiRand1K, MicroLens                 (new datasets, own vocab)
#   kuairec_first_average        (already has γ=0 — will skip automatically)
#   kuairec_first_individual     (newly requested)
#   kuairec_highest_average      (newly requested)
#   kuairec_highest_individual   (newly requested)
#
# Usage (ONE dataset at a time — single GPU):
#   nohup bash train_bestconfig_newds.sh 0 ML1M                      > best_ml1m.log 2>&1 &
#   nohup bash train_bestconfig_newds.sh 0 KuaiRand1K               > best_kui.log   2>&1 &
#   nohup bash train_bestconfig_newds.sh 0 MicroLens                > best_micro.log 2>&1 &
#   nohup bash train_bestconfig_newds.sh 0 kuairec_first_individual > best_fi.log    2>&1 &
#   nohup bash train_bestconfig_newds.sh 0 kuairec_highest_average  > best_ha.log    2>&1 &
#   nohup bash train_bestconfig_newds.sh 0 kuairec_highest_individual > best_hi.log  2>&1 &
# =============================================================================
set -u
GPU=${1:?Usage: train_bestconfig_newds.sh <GPU_ID> <DATASET>}
DS=${2:?DATASET must be one of: ML1M KuaiRand1K MicroLens kuairec_first_average kuairec_first_individual kuairec_highest_average kuairec_highest_individual}

case "$DS" in
  # --- New datasets (own vocab/cat counts, own vec file inside DIR) ---
  ML1M)       N=3126;  NCAT=18; DIR=./ML1M;       CATE=ml1m_cate.txt;       VEC=./ML1M/ml1m_vec.npy;        NEG="ML1M-random-sample_size=99-seed=4444.txt" ;;
  KuaiRand1K) N=20001; NCAT=44; DIR=./KuaiRand1K; CATE=kuairand_cate.txt;   VEC=./KuaiRand1K/kuairand_vec.npy;   NEG="KuaiRand-random-sample_size=99-seed=4444.txt" ;;
  MicroLens)  N=26923; NCAT=57; DIR=./MicroLens;  CATE=microlens_cate.txt;  VEC=./MicroLens/microlens_vec.npy;  NEG="MicroLens-random-sample_size=99-seed=4444.txt" ;;
  # --- KuaiRec variants (shared 10728 vocab + 31 cats, shared vec at KuaiRec_variants/) ---
  kuairec_first_average)        N=10728; NCAT=31; DIR=./KuaiRec_variants/${DS}; CATE=kuairec_cate.txt; VEC=./KuaiRec_variants/kuairec_vec.npy; NEG="KuaiRec-random-sample_size=99-seed=4444.txt" ;;
  kuairec_first_individual)     N=10728; NCAT=31; DIR=./KuaiRec_variants/${DS}; CATE=kuairec_cate.txt; VEC=./KuaiRec_variants/kuairec_vec.npy; NEG="KuaiRec-random-sample_size=99-seed=4444.txt" ;;
  kuairec_highest_average)      N=10728; NCAT=31; DIR=./KuaiRec_variants/${DS}; CATE=kuairec_cate.txt; VEC=./KuaiRec_variants/kuairec_vec.npy; NEG="KuaiRec-random-sample_size=99-seed=4444.txt" ;;
  kuairec_highest_individual)   N=10728; NCAT=31; DIR=./KuaiRec_variants/${DS}; CATE=kuairec_cate.txt; VEC=./KuaiRec_variants/kuairec_vec.npy; NEG="KuaiRec-random-sample_size=99-seed=4444.txt" ;;
  *) echo "Unknown DATASET: $DS"; exit 1 ;;
esac

RT_OUT="save_rt_fix_${DS}"
PT_OUT="save_pt_dense_lamb001_order0_${DS}"

echo "############################################################"
echo "# BEST-CONFIG RETRAIN (γ=0, dense PACER) — ${DS}"
echo "# vocab=${N}  n_cat=${NCAT}  GPU=${GPU}"
echo "# RT  -> ${RT_OUT}"
echo "# PT  -> ${PT_OUT}"
echo "############################################################"

# ---------------- Guard: data files ----------------
for f in "${DIR}/train-v0.txt" "${DIR}/valid-v0.txt" "${DIR}/test-v0.txt" \
         "${DIR}/${CATE}" "${VEC}" "${DIR}/${NEG}"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing data file $f"; exit 1
  fi
done
echo "[Guard] all data files present"

# ---------------- Stage 1: RT ----------------
lines=$(wc -l < "${RT_OUT}/train_result.txt" 2>/dev/null); lines=${lines:-0}
if [ -f "${RT_OUT}/DONE" ] || [ "$lines" -ge 1000 ]; then
  echo "[RT] ${RT_OUT} already complete (epoch ${lines}) — skipping"
else
  echo "[RT] training ${RT_OUT}"
  RESUME=""
  [ "$lines" -gt 0 ] && RESUME="-r"
  CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
      -tf ${DIR}/train-v0.txt -vf ${DIR}/valid-v0.txt -ef ${DIR}/test-v0.txt \
      -vn ${DIR}/${NEG} -en ${DIR}/${NEG} \
      -cat ${DIR}/${CATE} \
      -n ${N} -n_cat ${NCAT} -e 1000 -b 256 -l 1e-3 \
      -reg -t_mode topk -early_stop -patience 100 -min_delta 0.0001 \
      ${RESUME} -o ${RT_OUT} 2>&1 | tee rt_best_${DS}.log
  touch "${RT_OUT}/DONE"
fi

# ---------------- Stage 2: PT (γ=0, no L_order) ----------------
done_lines=$(wc -l < "${PT_OUT}/train_result.txt" 2>/dev/null); done_lines=${done_lines:-0}
if [ -f "${PT_OUT}/DONE" ] || [ -f "${PT_OUT}/test_result.txt" ] || [ "$done_lines" -ge 1000 ]; then
  echo "[PT] ${PT_OUT} already complete (DONE / epoch ${done_lines}) — skipping"
  exit 0
fi

# Resume reconcile
RESUME=""
NEWEST=$(ls "${PT_OUT}/model"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1)
if [ -n "$NEWEST" ]; then
  if [ "$done_lines" -lt "$NEWEST" ]; then
    while [ "$done_lines" -lt "$NEWEST" ]; do
      echo "recovered-epoch $((done_lines + 1))" >> "${PT_OUT}/train_result.txt"
      done_lines=$((done_lines + 1))
    done
  elif [ "$done_lines" -gt "$NEWEST" ]; then
    head -n "$NEWEST" "${PT_OUT}/train_result.txt" > "${PT_OUT}/train_result.txt.fix" \
      && mv "${PT_OUT}/train_result.txt.fix" "${PT_OUT}/train_result.txt"
  fi
  echo "[PT] resuming from epoch ${NEWEST}"
  RESUME="-r"
else
  echo "[PT] training ${PT_OUT} from scratch (γ=0, dense PACER)"
fi

CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
    -tf ${DIR}/train-v0.txt -vf ${DIR}/valid-v0.txt -ef ${DIR}/test-v0.txt \
    -vn ${DIR}/${NEG} -en ${DIR}/${NEG} \
    -cat ${DIR}/${CATE} -vec ${VEC} \
    -n ${N} -n_cat ${NCAT} -m train -e 1000 -b 256 -l 1e-3 \
    -dense -div -lamb 0.01 \
    -t_mode topk -early_stop -patience 100 -min_delta 0.0001 \
    ${RESUME} -i ./${RT_OUT} -o ./${PT_OUT} 2>&1 | tee pt_best_${DS}.log

echo "[PT] ${PT_OUT} finished"
