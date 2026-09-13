#!/bin/bash
# =============================================================================
# TEMPERATURE SWEEP — prospective intent softmax temperature tau_o
# (P_a(j) = softmax(F w_j / tau_o)). Everything is fixed:
#   variant = kuairec_first_average, TYPE family, dense supervision,
#   lambda = 0.01, batch 256, lr 1e-3, max epochs 1000, patience 100,
#   same RT checkpoint (save_rt_fix_kuairec_first_average).
# Only tau_o varies: 0.05, 0.2, 0.5, 1.0 are trained here.
# The tau_o = 0.1 cell (multiplier x10, original TRIER behavior) ALREADY
# EXISTS as save_pt_dense_lamb001_kuairec_first_average (note: dense naming
# uses lamb001 = -lamb 0.01; lamb01 would mean -lamb 0.1!) and is reused by
# the eval script — do not retrain it.
#
# Usage:
#   nohup bash train_temp_sweep_firstavg.sh <GPU_ID> > train_temp_sweep.log 2>&1 &
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=${1:-0}
MAX_EPOCHS=1000
VAR=kuairec_first_average
RT_DIR="save_rt_fix_${VAR}"

# TAG|tau_o
CONFIGS=(
    "tau005|0.05"
    "tau02|0.2"
    "tau05|0.5"
    "tau1|1.0"
)

echo "############################################################"
echo "# tau_o TEMPERATURE SWEEP training, GPU ${GPU}, ${VAR}"
echo "# (tau_o=0.1 is the existing save_pt_dense_lamb001_${VAR})"
echo "############################################################"

# Wait for RT (normally already long finished)
marker="${RT_DIR}/DONE"
rt_log="${RT_DIR}/train_result.txt"
lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
if [ ! -f "${marker}" ] && [ "${lines}" -lt "${MAX_EPOCHS}" ]; then
    echo "[Wait] RT not finished (epoch ${lines}/${MAX_EPOCHS}) - waiting..."
    while [ ! -f "${marker}" ]; do
        sleep 120
        lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
        [ "${lines}" -ge "${MAX_EPOCHS}" ] && break
        if ! pgrep -f "main_rt.py.*${VAR}" >/dev/null 2>&1; then
            echo "[Wait] RT stopped at epoch ${lines}"; break
        fi
    done
fi
echo "[Wait] RT complete (epoch $(wc -l < "${rt_log}" 2>/dev/null))."

for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG TAU <<< "$CFG"
    pt_dir="save_pt_dense_lamb001_${TAG}_${VAR}"
    pt_log="pt_temp_${TAG}_${VAR}.log"

    echo "============================================================"
    echo "PT tau_o=${TAU} -> ${pt_dir}"
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
        -dense -div -lamb 0.01 -tau_o ${TAU} \
        -t_mode topk -early_stop -patience 100 -min_delta 0.0001 \
        ${RESUME} \
        -i ./${RT_DIR} -o ./${pt_dir} 2>&1 | tee "${pt_log}"
done

echo "tau_o TEMPERATURE SWEEP TRAINING COMPLETE"
