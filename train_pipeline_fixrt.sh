#!/bin/bash
# =============================================================================
# TRIER Pipeline with FIXED RT semantics (paper-faithful retrospective RT)
#
# WHAT WAS FIXED (deviations from original chaoyushi/TRIER):
#   1. RT now trains on REVERSED sequences: given [sn..s2], predict s1 (first item).
#      Before: RT was a redundant forward model (predict last item) -> the
#      "left-side augmentation" actually generated FUTURE items and destroyed
#      recent context via torch.roll wraparound.
#   2. PT/Test datasets now feed REVERSED session[1:] to RT so beam search
#      generates genuine left-side (past) items.
#   3. RT regularizers (Dis_reg/ME_reg) now flow gradients (-reg), matching
#      the original implementation (previously no_grad + detach = no-op).
#   4. Hyperparameters restored to original TRIER defaults: -b 256 -l 1e-3
#      (previous runs used -b 64 -l 5e-4 = 8x smaller effective step).
#
# ALL NEW checkpoint dirs (old checkpoints preserved):
#   RT: save_rt_fix_<variant>
#   PT: save_pt_fixrt_<config>_<variant>
#
# Usage:
#   nohup bash train_pipeline_fixrt.sh > train_fixrt.log 2>&1 &
#   tail -f train_fixrt.log
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=0
MAX_EPOCHS=500

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

# SUFFIX|LAMB|LMD_CONSEC
CONFIGS=(
    "nodiv|0|0"
    "lamb0002|0.002|0"
    "lamb0005|0.005|0"
    "lamb001|0.01|0"
    "lamb005|0.05|0"
    "lamb01|0.1|0"
)

echo "############################################################"
echo "# FIXED-RT Pipeline: RT (retrospective, -reg) -> PT sweep"
echo "# $((${#CONFIGS[@]} * ${#VARIANTS[@]})) PT runs after 4 RT runs"
echo "# Hyperparams: -b 256 -l 1e-3 -e ${MAX_EPOCHS} (original defaults)"
echo "############################################################"
echo ""

# =============================================================================
# STAGE 1: Retrain RT models with corrected retrospective semantics
# =============================================================================
echo "[Stage 1] Training retrospective RT models (with -reg)"
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
        # Only resume if the checkpoint for the last completed epoch actually exists
        if [ -f "${rt_dir}/model/duorec-$((EPOCHS_DONE - 1)).pth" ]; then
            echo "  Resuming from epoch ${EPOCHS_DONE}"
            RESUME="-r"
        else
            echo "  Stale train_result.txt found (no checkpoint) - starting fresh"
            rm -f "${rt_dir}/train_result.txt"
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
        -early_stop -patience 50 -min_delta 0.0001 \
        ${RESUME} \
        -o ${rt_dir} 2>&1 | tee "${rt_log}"

    echo "  RT done: ${rt_dir}"
    echo ""
done

# =============================================================================
# STAGE 2: Train PT models (diversity sweep) with the fixed RT
# =============================================================================
echo "[Stage 2] Training PT models with fixed RT"
echo ""

for config_line in "${CONFIGS[@]}"; do
    IFS='|' read -r name lamb lmd_consec <<< "$config_line"

    for variant in "${VARIANTS[@]}"; do
        pt_dir="save_pt_fixrt_${name}_${variant}"
        pt_log="pt_fixrt_${name}_${variant}.log"
        rt_dir="save_rt_fix_${variant}"

        echo "============================================================"
        echo "PT Training: ${name} / ${variant} (lamb=${lamb}) -> ${pt_dir}"
        echo "============================================================"

        RESUME=""
        if [ -f "${pt_dir}/train_result.txt" ]; then
            EPOCHS_DONE=$(wc -l < "${pt_dir}/train_result.txt")
            if [ "$EPOCHS_DONE" -ge "$MAX_EPOCHS" ]; then
                echo "  Already complete (${EPOCHS_DONE} epochs) - skip"
                echo ""
                continue
            fi
            # Only resume if the checkpoint for the last completed epoch actually exists
            if [ -f "${pt_dir}/model/duorec-$((EPOCHS_DONE - 1)).pth" ]; then
                echo "  Resuming from epoch ${EPOCHS_DONE}"
                RESUME="-r"
            else
                echo "  Stale train_result.txt found (no checkpoint) - starting fresh"
                rm -f "${pt_dir}/train_result.txt"
            fi
        fi

        DIV_FLAGS=""
        [ "${lamb}" != "0" ] && DIV_FLAGS="-div -lamb ${lamb} -lmd_consec ${lmd_consec}"

        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${variant}/train-v0.txt \
            -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
            -ef ./KuaiRec_variants/${variant}/test-v0.txt \
            -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
            -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
            -m train -e ${MAX_EPOCHS} -b 256 -l 1e-3 \
            ${DIV_FLAGS} \
            -t_mode topk \
            -early_stop -patience 50 -min_delta 0.0001 \
            ${RESUME} \
            -i ./${rt_dir} \
            -o ./${pt_dir} 2>&1 | tee "${pt_log}"

        echo "  PT done: ${name} / ${variant}"
        echo ""
    done
done

echo "############################################################"
echo "FIXED-RT PIPELINE COMPLETE!"
echo "############################################################"
echo ""
echo "NOTE: RT valid/test metrics in rt_fix_*.log are NOT meaningful"
echo "(RT is now a retrospective model; the forward-prediction eval"
echo "task no longer matches its training task). RT quality shows up"
echo "in the PT results instead."
