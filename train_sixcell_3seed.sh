#!/bin/bash
# =============================================================================
# 3-SEED SIX-CELL ABLATION TRAINING (kuairec_first_average)
#
# Requirement (from paper reviewer): ablation table must report NDCG/Recall,
# ILD, CC, CS at ≥3 seeds, write mean±std. Four distinct configs × three seeds.
#
# FOUR DISTINCT CHECKPOINTS (each trained 3 seeds):
#   #  Cell(s)        flags                                  PT dir suffix
#   1  TRIER, TRIER-S -no_type -dense -div -lamb 0.01        notype_dense_lamb001_order0
#   2  TRIER-C       (type, dense) -div -lamb 0.01          dense_lamb001_order0
#   3  TRIER-L, PACER-LS -no_type + -soft_order_loss ...    notype_dense_lamb001_softo001
#   4  PACER-Full    (type) + -soft_order_loss ...           dense_lamb001_softo001
#
# OUTPUT DIRS: save_pt_<SUF>_kuairec_first_average_seed{1,2,3}/
#              plus the original non-suffixed dir is kept (from γ sweep;
#              eval will report mean±std across all four dirs — legacy + 3 seeds).
#
# RT: one shared checkpoint save_rt_fix_kuairec_first_average (frozen, same
#     across all seeds — RT is NOT part of the ablation).
#
# Usage:   bash train_sixcell_3seed.sh <GPU_ID>        # default seeds 1 2 3
#          SEEDS="2 42 7" bash train_sixcell_3seed.sh 0
#          MAX_EPOCHS=200 PATIENCE=20 bash train_sixcell_3seed.sh 0
# =============================================================================
set -u
GPU=${1:?Usage: train_sixcell_3seed.sh <GPU_ID>}
VAR=kuairec_first_average
DIR=./KuaiRec_variants/${VAR}
RT_OUT=save_rt_fix_${VAR}
N=10728; NCAT=31
CATE=kuairec_cate.txt; VEC=./KuaiRec_variants/kuairec_vec.npy
NEG="KuaiRec-random-sample_size=99-seed=4444.txt"

MAX_EPOCHS=${MAX_EPOCHS:-300}
PATIENCE=${PATIENCE:-30}
MIN_DELTA=${MIN_DELTA:-0.0001}
SEEDS=${SEEDS:-"1 2 3"}

# NAME|PT_SUF|TYPE_FLAG|L_ORDER_FLAGS
CONFIGS=(
  "base_notype|notype_dense_lamb001_order0|-no_type|"
  "base_type|dense_lamb001_order0||"
  "softo_notype|notype_dense_lamb001_softo001|-no_type|-soft_order_loss -soft_order_temp 1.0 -lmd_softorder 0.01"
  "softo_type|dense_lamb001_softo001||-soft_order_loss -soft_order_temp 1.0 -lmd_softorder 0.01"
)

echo "############################################################"
echo "# 3-SEED SIX-CELL TRAINING — ${VAR} (vocab ${N}, n_cat ${NCAT})"
echo "# seeds: ${SEEDS}"
echo "# MAX_EPOCHS=${MAX_EPOCHS}  PATIENCE=${PATIENCE}  MIN_DELTA=${MIN_DELTA}"
echo "# GPU=${GPU}"
echo "############################################################"

# Guard: data files
for f in "${DIR}/train-v0.txt" "${DIR}/valid-v0.txt" "${DIR}/test-v0.txt" \
         "${DIR}/${CATE}" "${VEC}" "${DIR}/${NEG}"; do
  [ -f "$f" ] || { echo "ERROR missing data: $f"; exit 1; }
done

# Stage 1: RT (shared, skip if DONE)
lines=$(wc -l < "${RT_OUT}/train_result.txt" 2>/dev/null); lines=${lines:-0}
if [ -f "${RT_OUT}/DONE" ] || [ "$lines" -ge "${MAX_EPOCHS}" ]; then
  echo "[RT] ${RT_OUT} complete (epoch ${lines}) — skip"
else
  RESUME=""; [ "$lines" -gt 0 ] && RESUME="-r"
  CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
      -tf ${DIR}/train-v0.txt -vf ${DIR}/valid-v0.txt -ef ${DIR}/test-v0.txt \
      -vn ${DIR}/${NEG} -en ${DIR}/${NEG} \
      -cat ${DIR}/${CATE} \
      -n ${N} -n_cat ${NCAT} -e ${MAX_EPOCHS} -b 256 -l 1e-3 \
      -reg -t_mode topk -early_stop -patience ${PATIENCE} -min_delta ${MIN_DELTA} \
      ${RESUME} -o ${RT_OUT} 2>&1 | tee rt_seed3.log
  touch "${RT_OUT}/DONE"
fi

# Stage 2: PT — 4 configs × N seeds
SEED_N=0
for SEED in ${SEEDS}; do
  SEED_N=$((SEED_N + 1))
  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r LABEL SUF TYPE_FLAG L_ORDER <<< "$CFG"
    OUT="save_pt_${SUF}_${VAR}_seed${SEED}"
    if [ -f "${OUT}/DONE" ] || [ -f "${OUT}/test_result.txt" ]; then
      echo "[PT seed${SEED}] ${OUT} complete — skip"
      continue
    fi

    # Resume reconcile
    RESUME=""
    NEWEST=$(ls "${OUT}/model"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1)
    if [ -n "$NEWEST" ]; then
      echo "[PT seed${SEED}] ${OUT} resuming from epoch ${NEWEST}"
      RESUME="-r"
    else
      echo "[PT seed${SEED}] ${OUT} training from scratch"
    fi

    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf ${DIR}/train-v0.txt -vf ${DIR}/valid-v0.txt -ef ${DIR}/test-v0.txt \
        -vn ${DIR}/${NEG} -en ${DIR}/${NEG} \
        -cat ${DIR}/${CATE} -vec ${VEC} \
        -n ${N} -n_cat ${NCAT} -m train -e ${MAX_EPOCHS} -b 256 -l 1e-3 \
        ${TYPE_FLAG} -dense -div -lamb 0.01 ${L_ORDER} \
        -t_mode topk -early_stop -patience ${PATIENCE} -min_delta ${MIN_DELTA} \
        -seed ${SEED} \
        ${RESUME} -i ./${RT_OUT} -o ./${OUT} 2>&1 | tee pt_${SUF}_seed${SEED}.log
  done
done

echo "############################################################"
echo "ALL 3-SEED SIX-CELL TRAINING DONE."
echo "DIRs: save_pt_*_kuairec_first_average_seed{1,2,3}/"
echo "Next: CUDA_VISIBLE_DEVICES=${GPU} bash eval_sixcell_3seed.sh"
echo "############################################################"
