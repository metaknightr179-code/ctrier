#!/bin/bash
# =============================================================================
# Train PT models with lamb=0.01 for all 4 Kuairec variants
# Reuses already-trained RT checkpoints from save_rt_*.
# Saves to save_pt_lamb001_* to keep results separate from lamb=0.1 models.
# Supports resuming: automatically detects existing checkpoints and resumes.
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
    local log_pattern="${save_dir#/./}_*.log"
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

FAILED_VARIANTS=""

echo "=============================================="
echo "Training PT models with lamb=0.01"
echo "(Overall diversity loss enabled, lambda=0.01)"
echo "=============================================="

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"
    PT_DIR="./save_pt_lamb001_${VAR}"

    echo "=============================================="
    echo "Variant: ${VAR}"
    echo "=============================================="

    # Verify RT checkpoint exists
    CKPT="${RT_DIR}/model/duorec-500.pth"
    if [ ! -f "$CKPT" ]; then
        # Try epoch 10 as fallback
        CKPT="${RT_DIR}/model/duorec-10.pth"
        if [ ! -f "$CKPT" ]; then
            echo "ERROR: Missing RT checkpoint for ${VAR}"
            echo "Skipping ${VAR}."
            continue
        fi
    fi

    # Check if training is complete or can resume
    PT_FLAGS=""
    if is_training_complete "$PT_DIR"; then
        echo "PT (lamb=0.01) already complete for ${VAR} — skipping"
        continue
    elif can_resume "$PT_DIR"; then
        EPOCHS_DONE=$(wc -l < "${PT_DIR}/train_result.txt")
        echo "Resuming PT (lamb=0.01) for ${VAR} from epoch ${EPOCHS_DONE}..."
        PT_FLAGS="-r"
    else
        echo "Training PT (lamb=0.01) for ${VAR} from scratch..."
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
        -div -lamb 0.01 -t_mode topk \
        -early_stop -patience 50 -min_delta 0.0001 \
        ${PT_FLAGS} \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee pt_lamb001_${VAR}.log
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "ERROR: Training failed for ${VAR} (exit code ${EXIT_CODE})"
        FAILED_VARIANTS="${FAILED_VARIANTS} ${VAR}"
    else
        echo "=== Variant ${VAR} complete ==="
    fi
    echo ""
done

echo "=============================================="
if [ -n "$FAILED_VARIANTS" ]; then
    echo "TRAINING FINISHED WITH FAILURES:"
    echo "Failed variants:${FAILED_VARIANTS}"
    echo "Check logs: pt_lamb001_<variant>.log"
else
    echo "ALL 4 VARIANTS (lamb=0.01) TRAINING COMPLETE!"
fi
echo "=============================================="
echo ""
echo "Output directories:"
for VAR in "${VARIANTS[@]}"; do
    echo "  PT (lamb=0.01): ./save_pt_lamb001_${VAR}/"
done
