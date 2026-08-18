#!/bin/bash
# =============================================================================
# Train PT models with multiple lambda values: 0.005, 0.01, 0.05, 0.1
# for all 4 Kuairec variants.
# Reuses already-trained RT checkpoints from save_rt_*.
# Supports resuming: automatically detects existing checkpoints and resumes.
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

# Lambda values and their directory suffixes
LAMBAS=("0.005" "0.01" "0.05" "0.1")
LAMB_SUFFIXES=("lamb0005" "lamb001" "lamb005" "lamb01")

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

FAILED=""

for LIDX in "${!LAMBAS[@]}"; do
    LAMB="${LAMBAS[$LIDX]}"
    SUFFIX="${LAMB_SUFFIXES[$LIDX]}"

    echo "############################################################"
    echo "# Training PT models with lamb=${LAMB}"
    echo "############################################################"
    echo ""

    for VAR in "${VARIANTS[@]}"; do
        RT_DIR="./save_rt_${VAR}"
        PT_DIR="./save_pt_${SUFFIX}_${VAR}"

        echo "=============================================="
        echo "Lambda=${LAMB}  Variant: ${VAR}"
        echo "=============================================="

        # Verify RT checkpoint exists
        CKPT="${RT_DIR}/model/duorec-500.pth"
        if [ ! -f "$CKPT" ]; then
            CKPT="${RT_DIR}/model/duorec-10.pth"
            if [ ! -f "$CKPT" ]; then
                echo "ERROR: Missing RT checkpoint for ${VAR}"
                echo "Skipping ${VAR}."
                FAILED="${FAILED} ${SUFFIX}/${VAR}"
                continue
            fi
        fi

        # Check if training is complete or can resume
        PT_FLAGS=""
        if is_training_complete "$PT_DIR"; then
            echo "PT (lamb=${LAMB}) already complete for ${VAR} — skipping"
            echo ""
            continue
        elif can_resume "$PT_DIR"; then
            EPOCHS_DONE=$(wc -l < "${PT_DIR}/train_result.txt")
            echo "Resuming PT (lamb=${LAMB}) for ${VAR} from epoch ${EPOCHS_DONE}..."
            PT_FLAGS="-r"
        else
            echo "Training PT (lamb=${LAMB}) for ${VAR} from scratch..."
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
            -div -lamb ${LAMB} -t_mode topk \
            -early_stop -patience 50 -min_delta 0.0001 \
            ${PT_FLAGS} \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee pt_${SUFFIX}_${VAR}.log
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        if [ "$EXIT_CODE" -ne 0 ]; then
            echo "ERROR: Training failed for lamb=${LAMB} ${VAR} (exit code ${EXIT_CODE})"
            FAILED="${FAILED} ${SUFFIX}/${VAR}"
        else
            echo "=== lamb=${LAMB} ${VAR} complete ==="
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
    echo "ALL LAMBDA x VARIANT COMBINATIONS COMPLETE!"
fi
echo "############################################################"
echo ""
echo "Output directories:"
for LIDX in "${!LAMBAS[@]}"; do
    SUFFIX="${LAMB_SUFFIXES[$LIDX]}"
    LAMB="${LAMBAS[$LIDX]}"
    for VAR in "${VARIANTS[@]}"; do
        echo "  PT (lamb=${LAMB}): ./save_pt_${SUFFIX}_${VAR}/"
    done
done
