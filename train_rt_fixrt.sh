#!/bin/bash
# =============================================================================
# Stage 1 of train_pipeline_fixrt.sh extracted as a standalone script:
# Retrospective RT training (reversed sequences, -reg, original hyperparams).
#
# Usage:
#   nohup bash train_rt_fixrt.sh <GPU_ID> > train_rt_1000.log 2>&1 &
#
# The PT stage scripts (train_pt_type_fixrt.sh / train_pt_notype_fixrt.sh)
# wait for main_rt.py processes to finish before starting.
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

echo "############################################################"
echo "# RT stage (retrospective, -reg), GPU ${GPU}"
echo "# Runs: ${#VARIANTS[@]}"
echo "# Hyperparams: -b 256 -l 1e-3 -e ${MAX_EPOCHS} -early_stop patience=100"
echo "############################################################"
echo ""

for variant in "${VARIANTS[@]}"; do
    rt_dir="save_rt_fix_${variant}"
    rt_log="rt_fix_${variant}.log"

    echo "=============================================="
    echo "RT Training -> ${rt_dir}"
    echo "=============================================="

    RESUME=""
    if [ -f "${rt_dir}/train_result.txt" ]; then
        EPOCHS_DONE=$(wc -l < "${rt_dir}/train_result.txt")
        if [ "$EPOCHS_DONE" -ge "$MAX_EPOCHS" ]; then
            echo "  Already complete (${EPOCHS_DONE} epochs) - skip"
            echo ""
            continue
        fi
        # main_rt.py -r loads duorec-<EPOCHS_DONE>.pth (one log line per epoch,
        # checkpoint number == completed epochs). On mismatch, abort instead of
        # deleting the log - never destroy training progress.
        if [ -f "${rt_dir}/model/duorec-${EPOCHS_DONE}.pth" ]; then
            echo "  Resuming from epoch ${EPOCHS_DONE}"
            RESUME="-r"
        else
            echo "  ERROR: ${EPOCHS_DONE} epochs logged but model/duorec-${EPOCHS_DONE}.pth missing - abort (progress preserved)."
            echo "  Recover manually: resume with -r -last_epoch <latest checkpoint number in ${rt_dir}/model/>"
            exit 1
        fi
    fi

    CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
        -tf ./KuaiRec_variants/${variant}/train-v0.txt \
        -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
        -ef ./KuaiRec_variants/${variant}/test-v0.txt \
        -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -e ${MAX_EPOCHS} -b 256 -l 1e-3 \
        -reg \
        -t_mode topk \
        -early_stop -patience 100 -min_delta 0.0001 \
        ${RESUME} \
        -o ${rt_dir} 2>&1 | tee "${rt_log}"

    echo "  RT done: ${rt_dir}"
    echo ""
done

echo "RT STAGE COMPLETE"
