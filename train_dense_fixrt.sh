#!/bin/bash
# =============================================================================
# Dense multi-position supervision training for TRIER PT.
#
# What: Instead of CE only on the last position, compute CE at every position
# (predict next item at each step). This matches GRU4Rec's BPTT training and
# gives ~50x more gradient signal per session.
#
# Only trains nodiv config (pure accuracy) for both type and notype families,
# on all 4 variants. Reuses existing RT checkpoints (RT has no dense mode).
#
# Checkpoint dirs use save_pt_dense_ prefix to avoid overwriting.
#
# Usage: nohup bash train_dense_fixrt.sh <GPU_ID> > train_dense.log 2>&1 &
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

# FAMILY|DIR_PREFIX|TYPE_FLAG
FAMILIES=(
    "type|save_pt_dense_|"
    "notype|save_pt_notype_dense_|-no_type"
)

echo "############################################################"
echo "# Dense multi-position supervision PT training, GPU ${GPU}"
echo "# Config: nodiv only (pure accuracy comparison)"
echo "# Hyperparams: -b 256 -l 1e-3 -e ${MAX_EPOCHS} -early_stop patience=100"
echo "############################################################"
echo ""

# Wait for RT to be complete (DONE marker or 1000-line log)
for variant in "${VARIANTS[@]}"; do
    marker="save_rt_fix_${variant}/DONE"
    rt_log="save_rt_fix_${variant}/train_result.txt"
    lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
    if [ ! -f "${marker}" ] && [ "${lines}" -lt "${MAX_EPOCHS}" ]; then
        echo "[Wait] RT for ${variant} not finished - waiting (poll every 120s)..."
        while [ ! -f "${marker}" ]; do
            sleep 120
            lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
            [ "${lines}" -ge "${MAX_EPOCHS}" ] && break
        done
    fi
    echo "[Wait] RT for ${variant} complete."
done

# Verify RT checkpoints exist
MISSING=0
for variant in "${VARIANTS[@]}"; do
    if ! ls "save_rt_fix_${variant}/model/"duorec-*.pth >/dev/null 2>&1; then
        echo "ERROR: no RT checkpoint in save_rt_fix_${variant}/model/ - abort."
        MISSING=1
    fi
done
if [ ${MISSING} -eq 1 ]; then
    echo "Train RT first: nohup bash train_rt_fixrt.sh <GPU_ID> > train_rt_1000.log 2>&1 &"
    exit 1
fi

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX TYPE_FLAG <<< "$FAM"
    for VAR in "${VARIANTS[@]}"; do
        pt_dir="${DIR_PREFIX}nodiv_${VAR}"
        pt_log="pt_dense_nodiv_${FAM_NAME}_${VAR}.log"
        rt_dir="save_rt_fix_${VAR}"
        VAR_DIR="./KuaiRec_variants/${VAR}"

        echo "============================================================"
        echo "PT Dense Training [${FAM_NAME}]: nodiv / ${VAR} -> ${pt_dir}"
        echo "============================================================"

        RESUME=""
        if [ -f "${pt_dir}/train_result.txt" ]; then
            EPOCHS_DONE=$(wc -l < "${pt_dir}/train_result.txt")
            if [ "$EPOCHS_DONE" -ge "$MAX_EPOCHS" ]; then
                echo "  Already complete (${EPOCHS_DONE} epochs) - skip"
                echo ""
                continue
            fi
            if [ -f "${pt_dir}/model/duorec-${EPOCHS_DONE}.pth" ]; then
                echo "  Resuming from epoch ${EPOCHS_DONE}"
                RESUME="-r"
            else
                echo "  ERROR: ${EPOCHS_DONE} epochs logged but checkpoint missing - abort."
                exit 1
            fi
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
            -dense \
            -t_mode topk \
            -early_stop -patience 100 -min_delta 0.0001 \
            ${RESUME} \
            ${TYPE_FLAG} \
            -i ./${rt_dir} \
            -o ./${pt_dir} 2>&1 | tee "${pt_log}"

        echo "  PT Dense [${FAM_NAME}] done: nodiv / ${VAR}"
        echo ""
    done
done

echo "DENSE PT TRAINING COMPLETE"
