#!/bin/bash
# =============================================================================
# Batch training script for all 4 Kuairec variants (single GPU)
# Trains RT then PT for each variant sequentially
# Supports resuming: automatically detects existing checkpoints and resumes
# =============================================================================

set -e

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

cd /root/ctrier

GPU=0
MAX_EPOCHS=500

# Check if a model has finished training (reached max epochs or early stopped)
# Args: $1 = save directory (e.g., ./save_rt_kuairec_highest_individual)
is_training_complete() {
    local save_dir="$1"
    local result_file="${save_dir}/train_result.txt"

    # No result file = never started
    if [ ! -f "$result_file" ]; then
        return 1
    fi

    local epoch_count
    epoch_count=$(wc -l < "$result_file")

    # Reached max epochs
    if [ "$epoch_count" -ge "$MAX_EPOCHS" ]; then
        return 0
    fi

    # Check for early stopping marker in log files
    # (Early stopping writes a break, so epoch_count < max but training is done)
    local log_pattern="${save_dir#/./}_*.log"
    if ls ${log_pattern} 1>/dev/null 2>&1; then
        if grep -q "EarlyStopping.*Loss converged" ${log_pattern} 2>/dev/null; then
            return 0
        fi
    fi

    return 1
}

# Check if training has started and can be resumed
# Args: $1 = save directory
can_resume() {
    local save_dir="$1"
    local result_file="${save_dir}/train_result.txt"

    if [ ! -f "$result_file" ]; then
        return 1
    fi

    local epoch_count
    epoch_count=$(wc -l < "$result_file")

    # Has at least 1 epoch logged
    if [ "$epoch_count" -ge 1 ]; then
        return 0
    fi

    return 1
}

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"
    PT_DIR="./save_pt_${VAR}"

    echo "=============================================="
    echo "Training variant: ${VAR}"
    echo "=============================================="

    # ------------------------------------------------------------------
    # Step 1: Train RT model
    # ------------------------------------------------------------------
    RT_FLAGS=""
    if is_training_complete "$RT_DIR"; then
        echo "[1/2] RT model already complete for ${VAR} — skipping"
    elif can_resume "$RT_DIR"; then
        EPOCHS_DONE=$(wc -l < "${RT_DIR}/train_result.txt")
        echo "[1/2] Resuming RT model for ${VAR} from epoch ${EPOCHS_DONE}..."
        RT_FLAGS="-r"
    else
        echo "[1/2] Training RT model for ${VAR} from scratch..."
    fi

    if [ -n "$RT_FLAGS" ] || ! is_training_complete "$RT_DIR"; then
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -e 500 -b 16 -l 5e-4 \
            -early_stop -patience 50 -min_delta 0.0001 \
            ${RT_FLAGS} \
            -o ${RT_DIR} 2>&1 | tee rt_${VAR}.log
    fi

    # Verify RT checkpoint exists (PT needs at least epoch 10)
    CKPT="${RT_DIR}/model/duorec-10.pth"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Missing RT checkpoint: $CKPT"
        echo "RT needs to train past epoch 10 before PT can start. Skipping ${VAR}."
        continue
    fi

    # ------------------------------------------------------------------
    # Step 2: Train PT model
    # ------------------------------------------------------------------
    PT_FLAGS=""
    if is_training_complete "$PT_DIR"; then
        echo "[2/2] PT model already complete for ${VAR} — skipping"
    elif can_resume "$PT_DIR"; then
        EPOCHS_DONE=$(wc -l < "${PT_DIR}/train_result.txt")
        echo "[2/2] Resuming PT model for ${VAR} from epoch ${EPOCHS_DONE}..."
        PT_FLAGS="-r"
    else
        echo "[2/2] Training PT model for ${VAR} from scratch..."
    fi

    if [ -n "$PT_FLAGS" ] || ! is_training_complete "$PT_DIR"; then
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -e 500 -b 16 -l 5e-4 \
            -div -lamb 0.1 -t_mode topk \
            -early_stop -patience 50 -min_delta 0.0001 \
            ${PT_FLAGS} \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee pt_${VAR}.log
    fi

    echo ""
    echo "=== Variant ${VAR} complete ==="
    echo ""
done

echo "=============================================="
echo "ALL 4 VARIANTS TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "Output directories:"
for VAR in "${VARIANTS[@]}"; do
    echo "  RT: ./save_rt_${VAR}/"
    echo "  PT: ./save_pt_${VAR}/"
done
