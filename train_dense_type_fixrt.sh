#!/bin/bash
# =============================================================================
# Dense PT training — TYPE family only
# Run in parallel with train_dense_notype_fixrt.sh on the same GPU
# Usage: nohup bash train_dense_type_fixrt.sh <GPU_ID> > train_dense_type.log 2>&1 &
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=${1:-0}
MAX_EPOCHS=1000

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

CONFIGS=(
    "nodiv|nodiv|"
    "lamb0002|lamb0002|-div -lamb 0.002"
    "lamb0005|lamb0005|-div -lamb 0.005"
    "lamb0005_consec0001|lamb0005_consec0001|-div -lamb 0.005 -lmd_consec 0.001"
    "lamb001|lamb001|-div -lamb 0.01"
    "lamb005|lamb005|-div -lamb 0.05"
    "lamb01|lamb01|-div -lamb 0.1"
)

echo "############################################################"
echo "# Dense PT [TYPE] training, GPU ${GPU}"
echo "############################################################"

# Wait for RT
for variant in "${VARIANTS[@]}"; do
    marker="save_rt_fix_${variant}/DONE"
    rt_log="save_rt_fix_${variant}/train_result.txt"
    lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
    if [ ! -f "${marker}" ] && [ "${lines}" -lt "${MAX_EPOCHS}" ]; then
        echo "[Wait] RT for ${variant} not finished (epoch ${lines}/${MAX_EPOCHS}) - waiting..."
        while [ ! -f "${marker}" ]; do
            sleep 120
            lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
            [ "${lines}" -ge "${MAX_EPOCHS}" ] && break
            if ! pgrep -f "main_rt.py.*${variant}" >/dev/null 2>&1; then
                echo "[Wait] RT for ${variant} stopped at epoch ${lines}"
                break
            fi
        done
    fi
    echo "[Wait] RT for ${variant} complete (epoch ${lines})."
done

for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r CFG_NAME CFG_SUFFIX CFG_FLAGS <<< "$CFG"
    for VAR in "${VARIANTS[@]}"; do
        pt_dir="save_pt_dense_${CFG_SUFFIX}_${VAR}"
        pt_log="pt_dense_${CFG_SUFFIX}_type_${VAR}.log"
        rt_dir="save_rt_fix_${VAR}"

        echo "============================================================"
        echo "PT Dense [TYPE]: ${CFG_NAME} / ${VAR} -> ${pt_dir}"
        echo "============================================================"

        # Completion check: DONE marker (written by main_pt on normal exit,
        # including early stop), eval results present, or ran to max epochs.
        # Early-stopped runs finish BELOW max epochs, so the old
        # "lines >= MAX_EPOCHS" test alone retried finished runs forever.
        EPOCHS_DONE=$(wc -l < "${pt_dir}/train_result.txt" 2>/dev/null); EPOCHS_DONE=${EPOCHS_DONE:-0}
        if [ -f "${pt_dir}/DONE" ] || [ -f "${pt_dir}/test_result.txt" ] || [ "$EPOCHS_DONE" -ge "$MAX_EPOCHS" ]; then
            echo "  Already complete (DONE / epoch ${EPOCHS_DONE}) - skip"
            continue
        fi

        # Resume from the NEWEST checkpoint, not from the log line count:
        # an aborted fresh start truncates train_result.txt ('w' mode) while the
        # keep-2 checkpoint cleanup leaves high-epoch checkpoints intact (it
        # deletes lowest-numbered). If the log fell behind the checkpoint epoch,
        # pad it so main_pt's line-count resume loads the right checkpoint.
        RESUME=""
        NEWEST=$(ls "${pt_dir}/model"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1)
        if [ -n "$NEWEST" ]; then
            if [ "$EPOCHS_DONE" -lt "$NEWEST" ]; then
                echo "  Repairing truncated log: ${EPOCHS_DONE} lines vs newest ckpt epoch ${NEWEST} (padding)"
                while [ "$EPOCHS_DONE" -lt "$NEWEST" ]; do
                    echo "recovered-epoch $((EPOCHS_DONE + 1))" >> "${pt_dir}/train_result.txt"
                    EPOCHS_DONE=$((EPOCHS_DONE + 1))
                done
            elif [ "$EPOCHS_DONE" -gt "$NEWEST" ]; then
                echo "  Repairing log ahead of ckpts: ${EPOCHS_DONE} lines vs newest ${NEWEST} (trimming)"
                head -n "$NEWEST" "${pt_dir}/train_result.txt" > "${pt_dir}/train_result.txt.fix" \
                    && mv "${pt_dir}/train_result.txt.fix" "${pt_dir}/train_result.txt"
            fi
            echo "  Resuming from epoch ${NEWEST}"
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
            ${RESUME} ${CFG_FLAGS} \
            -i ./${rt_dir} -o ./${pt_dir} 2>&1 | tee "${pt_log}"
    done
done

echo "DENSE PT [TYPE] TRAINING COMPLETE"
