#!/bin/bash
# =============================================================================
# Unified PT training script: trains ALL PT variants in one run.
#   - nodiv (lamb=0, lmd_consec=0)
#   - lamb=0.005, 0.01, 0.05, 0.1
#   - consec (lamb=0.01, lmd_consec=0.001)
# for all 4 Kuairec variants.
#
# Supports resuming: skips completed training, resumes partial training.
#
# Usage:
#   bash train_all.sh                  # train everything
#   nohup bash train_all.sh > train_all.log 2>&1 &   # background
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

# Each config: SUFFIX|LAMB|LMD_CONSEC|LABEL
CONFIGS=(
    "nodiv|0|0|No Diversity"
    "lamb0005|0.005|0|Lambda=0.005"
    "lamb001|0.01|0|Lambda=0.01"
    "lamb005|0.05|0|Lambda=0.05"
    "lamb01|0.1|0|Lambda=0.1"
    "consec0001|0.01|0.001|Consec (lamb=0.01, lmd_consec=0.001)"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

GPU=0
MAX_EPOCHS=500

is_training_complete() {
    local save_dir="$1"
    local result_file="${save_dir}/train_result.txt"
    if [ ! -f "$result_file" ]; then return 1; fi
    local epoch_count
    epoch_count=$(wc -l < "$result_file")
    if [ "$epoch_count" -ge "$MAX_EPOCHS" ]; then return 0; fi
    local log_pattern="pt_${save_dir#/./save_pt_}_*.log"
    if ls ${log_pattern} 1>/dev/null 2>&1; then
        if grep -q "EarlyStopping.*Loss converged" ${log_pattern} 2>/dev/null; then return 0; fi
    fi
    return 1
}

can_resume() {
    local save_dir="$1"
    local result_file="${save_dir}/train_result.txt"
    if [ ! -f "$result_file" ]; then return 1; fi
    local epoch_count
    epoch_count=$(wc -l < "$result_file")
    if [ "$epoch_count" -ge 1 ]; then return 0; fi
    return 1
}

FAILED=""

echo "############################################################"
echo "# Unified PT Training: All Configs x All Variants"
echo "# Configs: ${#CONFIGS[@]}"
echo "# Variants: ${#VARIANTS[@]}"
echo "# Total runs: $((${#CONFIGS[@]} * ${#VARIANTS[@]}))"
echo "############################################################"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX LAMB LMD_CONSEC LABEL <<< "$CONFIG"

    echo "############################################################"
    echo "# ${LABEL} (suffix: ${SUFFIX})"
    echo "############################################################"
    echo ""

    for VAR in "${VARIANTS[@]}"; do
        RT_DIR="./save_rt_${VAR}"
        PT_DIR="./save_pt_${SUFFIX}_${VAR}"

        echo "=============================================="
        echo "${LABEL} — ${VAR}"
        echo "=============================================="

        # Verify RT checkpoint exists
        CKPT="${RT_DIR}/model/duorec-500.pth"
        if [ ! -f "$CKPT" ]; then
            CKPT="${RT_DIR}/model/duorec-10.pth"
            if [ ! -f "$CKPT" ]; then
                echo "ERROR: Missing RT checkpoint for ${VAR}"
                FAILED="${FAILED} ${SUFFIX}/${VAR}"
                continue
            fi
        fi

        # Check if training is complete or can resume
        PT_FLAGS=""
        if is_training_complete "$PT_DIR"; then
            echo "Already complete — skipping"
            echo ""
            continue
        elif can_resume "$PT_DIR"; then
            EPOCHS_DONE=$(wc -l < "${PT_DIR}/train_result.txt")
            echo "Resuming from epoch ${EPOCHS_DONE}..."
            PT_FLAGS="-r"
        else
            echo "Training from scratch..."
        fi

        set +e
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -e 500 -b 16 -l 5e-4 \
            -div -lamb ${LAMB} -lmd_consec ${LMD_CONSEC} \
            -t_mode topk \
            -early_stop -patience 50 -min_delta 0.0001 \
            ${PT_FLAGS} \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee pt_${SUFFIX}_${VAR}.log
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        if [ "$EXIT_CODE" -ne 0 ]; then
            echo "ERROR: Training failed for ${SUFFIX}/${VAR} (exit code ${EXIT_CODE})"
            FAILED="${FAILED} ${SUFFIX}/${VAR}"
        else
            echo "=== ${SUFFIX}/${VAR} complete ==="
        fi
        echo ""
    done
done

echo "############################################################"
if [ -n "$FAILED" ]; then
    echo "TRAINING FINISHED WITH FAILURES:"
    echo "${FAILED}"
    echo "Check corresponding pt_<suffix>_<variant>.log files"
else
    echo "ALL TRAINING COMPLETE!"
fi
echo "############################################################"
echo ""
echo "Output directories:"
for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX LAMB LMD_CONSEC LABEL <<< "$CONFIG"
    for VAR in "${VARIANTS[@]}"; do
        echo "  ./save_pt_${SUFFIX}_${VAR}/"
    done
done