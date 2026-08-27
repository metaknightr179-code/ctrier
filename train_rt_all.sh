#!/bin/bash
# =============================================================================
# Unified RT training script: trains RT (Reverse Trajectory) models for all
# 4 KuaiRec variants.
#
# RT model provides left-side augmentation for PT training.
# Must be trained BEFORE PT (PT depends on RT checkpoints).
#
# Usage:
#   bash train_rt_all.sh                  # train all RT models
#   nohup bash train_rt_all.sh > train_rt_all.log 2>&1 &   # background
#   bash train_rt_all.sh -r               # resume incomplete training
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
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
    local log_pattern="rt_${save_dir#/./save_rt_}_*.log"
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

RESUME_FLAG=""
if [ "$1" == "-r" ]; then
    RESUME_FLAG="-r"
fi

FAILED=""

echo "############################################################"
echo "# Unified RT Training: All KuaiRec Variants"
echo "# Variants: ${#VARIANTS[@]}"
echo "############################################################"
echo ""

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"

    echo "=============================================="
    echo "RT Training — ${VAR}"
    echo "=============================================="

    # Check if training is complete or can resume
    RT_FLAGS=""
    if is_training_complete "$RT_DIR"; then
        echo "Already complete — skipping"
        echo ""
        continue
    elif can_resume "$RT_DIR"; then
        EPOCHS_DONE=$(wc -l < "${RT_DIR}/train_result.txt")
        echo "Resuming from epoch ${EPOCHS_DONE}..."
        RT_FLAGS="-r"
    else
        echo "Training from scratch..."
    fi

    set +e
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
        -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
        -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
        -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
        -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -e 500 -b 64 -l 5e-4 \
        -early_stop -patience 50 -min_delta 0.0001 \
        ${RT_FLAGS} \
        -o ${RT_DIR} 2>&1 | tee rt_${VAR}.log
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "ERROR: Training failed for ${VAR} (exit code ${EXIT_CODE})"
        FAILED="${FAILED} ${VAR}"
    else
        echo "=== RT ${VAR} complete ==="
    fi
    echo ""
done

echo "############################################################"
if [ -n "$FAILED" ]; then
    echo "RT TRAINING FINISHED WITH FAILURES:"
    echo "${FAILED}"
    echo "Check corresponding rt_*.log files"
else
    echo "ALL RT TRAINING COMPLETE!"
fi
echo "############################################################"
echo ""
echo "RT checkpoint directories:"
for VAR in "${VARIANTS[@]}"; do
    echo "  ./save_rt_${VAR}/"
done
