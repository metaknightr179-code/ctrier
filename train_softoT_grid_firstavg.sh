#!/bin/bash
# =============================================================================
# T (SOFT-SELECTION TEMPERATURE) GRID for the FIXED L_order —
# kuairec_first_average, dense, lambda=0.01, fixed gamma_o = 0.01.
#
# This sweeps T in  pi_s = Q_s^(1/T) / Z  (see trier_pt.py soft_order_loss;
# the double-softmax bug fixed 2026-09-15, commit 78dcbfe). It is NOT the
# prospective-intent tau_o sweep (train_temp_sweep_firstavg.sh): tau_o shapes
# P_cov in the decoder, T sharpens the loss's soft item selection.
#
#   T=1.0   pi = the decoder's own mixture (natural default; reused from the
#           gamma grid dir save_pt_{notype_}dense_lamb001_softo001_<variant>,
#           train it via train_consecgamma_grid_firstavg.sh if missing)
#   2.0     flatter than the decoder distribution (approach the broken regime)
#   0.5     squared  (sharper)
#   0.25    fourth power
#   0.1     tenth power (near-hard argmax; gradients get spiky)
#
# Newly trained here (4 points x 2 families = 8 runs):
#   T 2.0  -> save_pt_{notype_}dense_lamb001_softo001_T2_<variant>
#   T 0.5  -> save_pt_{notype_}dense_lamb001_softo001_T05_<variant>
#   T 0.25 -> save_pt_{notype_}dense_lamb001_softo001_T025_<variant>
#   T 0.1  -> save_pt_{notype_}dense_lamb001_softo001_T01_<variant>
#
# Everything else is fixed: -soft_order_loss -lmd_softorder 0.01, dense CE,
# t_mode topk, b=256, lr=1e-3, 1000 epochs, patience 100, frozen RT.
# (Dir names encode gamma=0.01 — the T sweep intentionally varies exactly one
# hyperparameter. For another gamma copy this script and rename dirs.)
#
# Knobs:
#   ST_CONFIGS="T05|0.5 T025|0.25"  ... T subset
#   ST_ONLY=type|notype          ... one family (2-GPU split)
#     ST_ONLY=type   nohup bash train_softoT_grid_firstavg.sh 0 > st_t.log 2>&1 &
#     ST_ONLY=notype nohup bash train_softoT_grid_firstavg.sh 1 > st_n.log 2>&1 &
#
# Usage:
#   nohup bash train_softoT_grid_firstavg.sh <GPU_ID> > train_softoT.log 2>&1 &
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=${1:-0}
MAX_EPOCHS=${MAX_EPOCHS:-1000}
PATIENCE=${PATIENCE:-100}
GAMMA=0.01            # fixed: the T sweep varies exactly one hyperparameter
VARIANTS=( ${ST_VARIANTS:-kuairec_first_average} )

# FAMILY|DIR_MIDDLE|TYPE_FLAG
case "${ST_ONLY:-}" in
    type)   FAMILIES=( "type||" ) ;;
    notype) FAMILIES=( "notype|notype_|-no_type" ) ;;
    *)      FAMILIES=( "type||" "notype|notype_|-no_type" ) ;;
esac

# SUFFIX|T   (T=1.0 is NOT here: reused from lamb001_softo001)
if [ -n "${ST_CONFIGS:-}" ]; then
    CONFIGS=( $ST_CONFIGS )
else
    CONFIGS=( "T2|2.0" "T05|0.5" "T025|0.25" "T01|0.1" )
fi

echo "############################################################"
echo "# SOFT-SELECTION T grid at lambda=0.01, gamma_o=${GAMMA}, GPU ${GPU}"
echo "# variants: ${VARIANTS[*]}; T points: ${#CONFIGS[@]} x ${#FAMILIES[@]} families"
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
    IFS='|' read -r SUFFIX TEMP <<< "$CFG"
    for VAR in "${VARIANTS[@]}"; do
        pt_dir="save_pt_${DIR_MID}dense_lamb001_softo001_${SUFFIX}_${VAR}"
        pt_log="pt_dense_lamb001_softo001_${SUFFIX}_${FAM_NAME}_${VAR}.log"
        rt_dir="save_rt_fix_${VAR}"

        echo "============================================================"
        echo "PT Dense [${FAM_NAME}]: softo001_${SUFFIX} (T=${TEMP}, gamma_o=${GAMMA}) / ${VAR}"
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
            -dense -t_mode topk -early_stop -patience ${PATIENCE} -min_delta 0.0001 \
            ${RESUME} ${TYPE_FLAG} \
            -div -lamb 0.01 \
            -soft_order_loss -soft_order_temp ${TEMP} -lmd_softorder ${GAMMA} \
            -i ./${rt_dir} -o ./${pt_dir} 2>&1 | tee "${pt_log}"
    done
  done
done

echo "SOFT-SELECTION T GRID TRAINING DONE"
