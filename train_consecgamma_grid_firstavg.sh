#!/bin/bash
# =============================================================================
# gamma_consec TRAINING-LOSS-WEIGHT GRID — kuairec_first_average, dense,
# fixed lambda=0.005.
#
# NAMING HISTORY (important):
#   Before 2026-09-11 the loss weight was the flag -lmd_consec (default .01);
#   commit a3225a7 RENAMED it to -gamma_consec and gave -lmd_consec to a NEW
#   inference/score-time cosine penalty. The pre-existing checkpoints
#     save_pt_{notype_}dense_lamb0005_consec0001_<variant>  (gamma = 0.001)
#     save_pt_{notype_}dense_lamb0005_consec005_<variant>   (gamma = 0.05)
#     save_pt_{notype_}dense_lamb0005_consec01_<variant>    (gamma = 0.1)
#   were trained Sep 7-9 with the OLD flag, i.e. they ARE the gamma grid
#   points — do NOT delete or move them.
#
# Grid gamma_consec in {0, 0.001, 0.005, 0.01, 0.05, 0.1}:
#   0     -> save_pt_{notype_}dense_lamb0005_consec0_<variant>     (NEW)
#   0.001 -> save_pt_{notype_}dense_lamb0005_consec0001_<variant>  (exists)
#   0.005 -> save_pt_{notype_}dense_lamb0005_consec0005_<variant>  (NEW)
#   0.01  -> plain save_pt_{notype_}dense_lamb0005_<variant>       (exists;
#            trained with the default gamma_consec = 0.01)
#   0.05  -> save_pt_{notype_}dense_lamb0005_consec005_<variant>   (exists)
#   0.1   -> save_pt_{notype_}dense_lamb0005_consec01_<variant>    (exists)
#
# This script trains ONLY the two missing points (0 and 0.005), type + notype,
# i.e. 4 runs. Everything else is fixed: dense CE, t_mode topk, b=256,
# lr=1e-3, 1000 epochs, patience 100, frozen RT. The new SCORE penalty
# (-lmd_consec) is left at its default 0 — it is not part of this sweep.
#
# Usage:
#   nohup bash train_consecgamma_grid_firstavg.sh <GPU_ID> > train_cgamma.log 2>&1 &
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=${1:-0}
MAX_EPOCHS=1000
VARIANTS=( ${CG_VARIANTS:-kuairec_first_average} )

# FAMILY|DIR_MIDDLE|TYPE_FLAG
FAMILIES=(
    "type||"
    "notype|notype_|-no_type"
)

# SUFFIX|gamma_consec
CONFIGS=(
    "consec0|0"
    "consec0005|0.005"
)

echo "############################################################"
echo "# gamma_consec loss-weight grid (fixed lambda=0.005), GPU ${GPU}"
echo "# variants: ${VARIANTS[*]}; training missing points: 0, 0.005"
echo "############################################################"

for VAR in "${VARIANTS[@]}"; do
  RT_DIR="save_rt_fix_${VAR}"
  marker="${RT_DIR}/DONE"
  rt_log="${RT_DIR}/train_result.txt"
  if [ -f "$rt_log" ]; then lines=$(wc -l < "${rt_log}"); else lines=0; fi
  if [ ! -f "${marker}" ] && [ "${lines}" -lt "${MAX_EPOCHS}" ]; then
      echo "[Wait] RT for ${VAR} not finished (epoch ${lines}/${MAX_EPOCHS}) - waiting..."
      while [ ! -f "${marker}" ]; do
          sleep 120
          if [ -f "$rt_log" ]; then lines=$(wc -l < "${rt_log}"); else lines=0; fi
          [ "${lines}" -ge "${MAX_EPOCHS}" ] && break
          if ! pgrep -f "main_rt.py.*${VAR}" >/dev/null 2>&1; then
              echo "[Wait] RT for ${VAR} stopped at epoch ${lines}"; break
          fi
      done
  fi
  echo "[Wait] RT for ${VAR} complete (epoch ${lines})."
done

for FAM in "${FAMILIES[@]}"; do
  IFS='|' read -r FAM_NAME DIR_MID TYPE_FLAG <<< "$FAM"
  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX GAMMA <<< "$CFG"
    for VAR in "${VARIANTS[@]}"; do
        pt_dir="save_pt_${DIR_MID}dense_lamb0005_${SUFFIX}_${VAR}"
        pt_log="pt_dense_lamb0005_${SUFFIX}_${FAM_NAME}_${VAR}.log"
        rt_dir="save_rt_fix_${VAR}"

        echo "============================================================"
        echo "PT Dense [${FAM_NAME}]: lamb0005_${SUFFIX} (gamma_consec=${GAMMA}) / ${VAR}"
        echo "-> ${pt_dir}"
        echo "============================================================"

        if [ -f "${pt_dir}/train_result.txt" ]; then
            EPOCHS_DONE=$(wc -l < "${pt_dir}/train_result.txt")
        else
            EPOCHS_DONE=0
        fi
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
            -div -lamb 0.005 -gamma_consec ${GAMMA} \
            -i ./${rt_dir} -o ./${pt_dir} 2>&1 | tee "${pt_log}"
    done
  done
done

echo "gamma_consec GRID TRAINING DONE"
