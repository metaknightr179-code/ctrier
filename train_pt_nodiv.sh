#!/bin/bash
# =============================================================================
# Train PT models with ALL diversity losses set to zero (lamb=0, lmd_consec=0)
# for all 4 Kuairec variants. This serves as the TRIER baseline without diversity.
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
    local log_pattern="pt_nodiv_*.log"
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
echo "# PT Training: lamb=0  lmd_consec=0 (no diversity)"
echo "############################################################"
echo ""

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"
    PT_DIR="./save_pt_nodiv_${VAR}"

    echo "=============================================="
    echo "Variant: ${VAR}  (lamb=0, lmd_consec=0)"
    echo "=============================================="

    # Verify RT checkpoint exists
    CKPT="${RT_DIR}/model/duorec-500.pth"
    if [ ! -f "$CKPT" ]; then
        CKPT="${RT_DIR}/model/duorec-10.pth"
        if [ ! -f "$CKPT" ]; then
            echo "ERROR: Missing RT checkpoint for ${VAR}"
            FAILED="${FAILED} nodiv/${VAR}"
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
        -div -lamb 0 \
        -lmd_consec 0 \
        -t_mode topk \
        -early_stop -patience 50 -min_delta 0.0001 \
        ${PT_FLAGS} \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee pt_nodiv_${VAR}.log
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "ERROR: Training failed for ${VAR} (exit code ${EXIT_CODE})"
        FAILED="${FAILED} nodiv/${VAR}"
    else
        echo "=== ${VAR} complete ==="
    fi
    echo ""
done

echo "############################################################"
if [ -n "$FAILED" ]; then
    echo "TRAINING FINISHED WITH FAILURES:"
    echo "${FAILED}"
else
    echo "ALL VARIANTS COMPLETE!"
fi
echo "############################################################"
echo ""
echo "Output directories:"
for VAR in "${VARIANTS[@]}"; do
    echo "  ./save_pt_nodiv_${VAR}/"
done
