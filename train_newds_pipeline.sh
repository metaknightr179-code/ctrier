#!/bin/bash
# Pipeline for the new datasets: RT first, then the full PT grid (7 configs x type/notype).
#
# Usage:   bash train_newds_pipeline.sh <GPU_ID> <DATASET>
#          DATASET in {ML1M, KuaiRand1K, MicroLens}
# DENSE=0  bash train_newds_pipeline.sh 0 ML1M     # official-TRIER variant (no dense supervision)
#
# - Skips RT if already done (DONE marker or train_result.txt >= 1000 lines), resumes if partial.
# - Skips PT configs whose train_result.txt already exists.
# - Safe to launch all three datasets in parallel on one GPU (small models).

set -u
GPU=${1:?Usage: train_newds_pipeline.sh <GPU_ID> <DATASET>}
DS=${2:?DATASET must be ML1M, KuaiRand1K or MicroLens}
DENSE_FLAG="-dense"
[ "${DENSE:-1}" = "0" ] && DENSE_FLAG=""

case "$DS" in
  ML1M)
    N=3126; NCAT=18; DIR=./ML1M; CATE=ml1m_cate.txt; VEC=ml1m_vec.npy
    NEG="ML1M-random-sample_size=99-seed=4444.txt" ;;
  KuaiRand1K)
    N=133868; NCAT=49; DIR=./KuaiRand1K; CATE=kuairand_cate.txt; VEC=kuairand_vec.npy
    NEG="KuaiRand-random-sample_size=99-seed=4444.txt" ;;
  MicroLens)
    N=26923; NCAT=57; DIR=./MicroLens; CATE=microlens_cate.txt; VEC=microlens_vec.npy
    NEG="MicroLens-random-sample_size=99-seed=4444.txt" ;;
  *) echo "Unknown DATASET: $DS"; exit 1 ;;
esac

# 7 configs: name|extra_flags   (mirrors the KuaiRec dense grid)
CONFIGS=(
  "nodiv|"
  "lamb0002|-div -lamb 0.002"
  "lamb0005|-div -lamb 0.005"
  "lamb0005_consec0001|-div -lamb 0.005 -lmd_consec 0.001"
  "lamb001|-div -lamb 0.01"
  "lamb005|-div -lamb 0.05"
  "lamb01|-div -lamb 0.1"
)

RT_OUT="save_rt_fix_${DS}"
export CUDA_VISIBLE_DEVICES=${GPU}

# ---------------- Stage 1: RT (retrospective) ----------------
lines=$(wc -l < "${RT_OUT}/train_result.txt" 2>/dev/null); lines=${lines:-0}
if [ -f "${RT_OUT}/DONE" ] || [ "$lines" -ge 1000 ]; then
  echo "[RT] ${RT_OUT} already complete (epoch ${lines}) - skipping"
else
  echo "[RT] training ${RT_OUT}"
  RESUME=""
  [ "$lines" -gt 0 ] && RESUME="-r"
  python3 main_rt.py \
    -tf ${DIR}/train-v0.txt -vf ${DIR}/valid-v0.txt -ef ${DIR}/test-v0.txt \
    -vn ${DIR}/${NEG} -en ${DIR}/${NEG} \
    -cat ${DIR}/${CATE} \
    -n ${N} -n_cat ${NCAT} -e 1000 -b 256 -l 1e-3 \
    -reg -t_mode topk -early_stop -patience 100 -min_delta 0.0001 \
    ${RESUME} -o ${RT_OUT} 2>&1 | tee rt_${DS}.log
  touch "${RT_OUT}/DONE"
fi

# ---------------- Stage 2: PT grid (type + notype) ----------------
for family in type notype; do
  if [ "$family" = "notype" ]; then FAM_FLAG="-no_type"; PREFIX="save_pt_notype_dense"; else FAM_FLAG=""; PREFIX="save_pt_dense"; fi
  for cfg in "${CONFIGS[@]}"; do
    name="${cfg%%|*}"; extra="${cfg#*|}"
    OUT="${PREFIX}_${name}_${DS}"
    [ -n "$DENSE_FLAG" ] || OUT="${OUT/_dense/}"   # DENSE=0 -> drop _dense from dir name
    done_lines=$(wc -l < "${OUT}/train_result.txt" 2>/dev/null); done_lines=${done_lines:-0}
    if [ "$done_lines" -ge 1000 ]; then
      echo "[PT ${family}] ${OUT} already complete - skipping"
      continue
    fi
    echo "[PT ${family}] training ${OUT}"
    RESUME=""
    [ "$done_lines" -gt 0 ] && RESUME="-r"
    python3 main_pt.py \
      -tf ${DIR}/train-v0.txt -vf ${DIR}/valid-v0.txt -ef ${DIR}/test-v0.txt \
      -vn ${DIR}/${NEG} -en ${DIR}/${NEG} \
      -cat ${DIR}/${CATE} -vec ${DIR}/${VEC} \
      -n ${N} -n_cat ${NCAT} -m train -e 1000 -b 256 -l 1e-3 \
      ${DENSE_FLAG} ${FAM_FLAG} ${extra} \
      -t_mode topk -early_stop -patience 100 -min_delta 0.0001 \
      ${RESUME} -i ./${RT_OUT} -o ./${OUT} 2>&1 | tee pt_${PREFIX}_${name}_${DS}.log
  done
done

echo "PIPELINE [${DS}] COMPLETE"
