#!/bin/bash
# =============================================================================
# lambda_c GRID COMPLETION — TRAIN the two missing score-penalty points.
#
# The established lambda_c sweep (see existing checkpoint dirs) consists of
# DENSE checkpoints trained at fixed lambda=0.005 with -lmd_consec baked into
# the generation score used to select the diverse training tokens:
#
#   lambda_c = 0     save_pt_{notype_}dense_lamb0005_<variant>          (exists)
#   lambda_c = 0.001 save_pt_{notype_}dense_lamb0005_consec0001_<variant> (exists)
#   lambda_c = 0.005 save_pt_{notype_}dense_lamb0005_consec0005_<variant> (NEW)
#   lambda_c = 0.01  save_pt_{notype_}dense_lamb0005_consec001_<variant>  (NEW)
#   lambda_c = 0.05  save_pt_{notype_}dense_lamb0005_consec005_<variant>  (exists)
#   lambda_c = 0.1   save_pt_{notype_}dense_lamb0005_consec01_<variant>   (exists)
#
# Suffix rule: the decimal point of lambda_c is deleted (0.005 -> consec0005).
# This script trains only the 2 missing points, for both TYPE (PACER Full) and
# NOTYPE families, on kuairec_first_average by default (override with the
# LMC_VARIANTS env var, e.g. LMC_VARIANTS="kuairec_first_average ...").
# Everything else is identical to train_dense_type_fixrt.sh /
# train_dense_notype_fixrt.sh: dense CE, t_mode topk, b=256, lr=1e-3,
# 1000 epochs, early stop patience 100, same frozen RT.
#
# Usage:
#   nohup bash train_lmdconsec_grid_firstavg.sh <GPU_ID> > train_lmc.log 2>&1 &
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=${1:-0}
MAX_EPOCHS=1000
VARIANTS=( ${LMC_VARIANTS:-kuairec_first_average} )

# FAMILY|DIR_MIDDLE|TYPE_FLAG
FAMILIES=(
    "type||"
    "notype|notype_|-no_type"
)

# SUFFIX|lambda_c
CONFIGS=(
    "consec0005|0.005"
    "consec001|0.01"
)

echo "############################################################"
echo "# lambda_c grid completion (fixed lambda=0.005), GPU ${GPU}"
echo "# variants: ${VARIANTS[*]}; families: type, notype"
echo "############################################################"

for VAR in "${VARIANTS[@]}"; do
  RT_DIR="save_rt_fix_${VAR}"
  marker="${RT_DIR}/DONE"
  rt_log="${RT_DIR}/train_result.txt"
  lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
  if [ ! -f "${marker}" ] && [ "${lines}" -lt "${MAX_EPOCHS}" ]; then
      echo "[Wait] RT for ${VAR} not finished (epoch ${lines}/${MAX_EPOCHS}) - waiting..."
      while [ ! -f "${marker}" ]; do
          sleep 120
          lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
          [ "${lines}" -ge "${MAX_EPOCHS}" ] && break
          if ! pgrep -f "main_rt.py.*${VAR}" >/dev/null 2>&1; then
              echo "[Wait] RT for ${VAR} stopped at epoch ${lines}"; break
          fi
      done
  fi
  echo "[Wait] RT for ${VAR} complete (epoch $(wc -l < "${rt_log}" 2>/dev/null))."
done

for FAM in "${FAMILIES[@]}"; do
  IFS='|' read -r FAM_NAME DIR_MID TYPE_FLAG <<< "$FAM"
  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX LMC <<< "$CFG"
    for VAR in "${VARIANTS[@]}"; do
        pt_dir="save_pt_${DIR_MID}dense_lamb0005_${SUFFIX}_${VAR}"
        pt_log="pt_dense_lamb0005_${SUFFIX}_${FAM_NAME}_${VAR}.log"
        rt_dir="save_rt_fix_${VAR}"

        echo "============================================================"
        echo "PT Dense [${FAM_NAME}]: lamb0005_${SUFFIX} (lmd_consec=${LMC}) / ${VAR}"
        echo "-> ${pt_dir}"
        echo "============================================================"

        EPOCHS_DONE=$(wc -l < "${pt_dir}/train_result.txt" 2>/dev/null); EPOCHS_DONE=${EPOCHS_DONE:-0}
        if [ -f "${pt_dir}/DONE" ] || [ -f "${pt_dir}/test_result.txt" ] || [ "$EPOCHS_DONE" -ge "$MAX_EPOCHS" ]; then
            echo "  Already complete (DONE / epoch ${EPOCHS_DONE}) - skip"
            continue
        fi

        RESUME=""
        NEWEST=$(ls "${pt_dir}/model"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1)
        if [ -n "$NEWEST" ]; then
            if [ "$EPOCHS_DONE" -lt "$NEWEST" ]; then
                while [ "$EPOCHS_DONE" -lt "$NEWEST" ]; do
                    echo "recovered-epoch $((EPOCHS_DONE + 1))" >> "${pt_dir}/train_result.txt"
                    EPOCHS_DONE=$((EPOCHS_DONE + 1))
                done
            elif [ "$EPOCHS_DONE" -gt "$NEWEST" ]; then
                head -n "$NEWEST" "${pt_dir}/train_result.txt" > "${pt_dir}/train_result.txt.fix" \
                    && mv "${pt_dir}/train_result.txt.fix" "${pt_dir}/train_result.txt"
            fi
            RESUME="-r"
        fi

        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
            -m train -e ${MAX_EPOCHS} -b 256 -l 1e-3 \
            -dense -t_mode topk -early_stop -patience 100 -min_delta 0.0001 \
            ${RESUME} ${TYPE_FLAG} \
            -div -lamb 0.005 -lmd_consec ${LMC} \
            -i ./${rt_dir} -o ./${pt_dir} 2>&1 | tee "${pt_log}"
    done
  done
done

echo "lambda_c GRID COMPLETION TRAINING DONE"
