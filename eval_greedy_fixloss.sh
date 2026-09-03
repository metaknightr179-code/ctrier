#!/bin/bash
# =============================================================================
# Evaluate existing fixloss checkpoints with GREEDY mode (no retraining needed!)
#
# t_mode is inference-only — training is identical regardless of t_mode.
# This script re-evaluates all existing fixloss checkpoints with -t_mode greedy
# instead of -t_mode topk, so we can compare the two inference strategies.
#
# Greedy mode: applies score = lamb * div_score + (1 - lamb) * rel_score
#             at inference time (step-by-step generation with RT model)
# Topk mode:  pure relevance ranking (no diversity at inference)
#
# Usage:
#   bash eval_greedy_fixloss.sh
# =============================================================================

cd "$(dirname "$0")"

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

# SUFFIX|LAMB|LMD_CONSEC|USE_DIV (1=use -div flag, 0=don't)
CONFIGS=(
    "fixloss_nodiv|0|0|0"
    "fixloss_lamb0002|0.002|0|1"
    "fixloss_lamb0005|0.005|0|1"
    "fixloss_lamb001|0.01|0|1"
    "fixloss_lamb005|0.05|0|1"
    "fixloss_lamb01|0.1|0|1"
)

get_latest_epoch() {
    ls "${1}/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

echo "############################################################"
echo "# Greedy-mode evaluation of fixloss checkpoints"
echo "# (t_mode is inference-only — no retraining needed)"
echo "############################################################"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX LAMB LMD_CONSEC USE_DIV <<< "$CONFIG"

    for VAR in "${VARIANTS[@]}"; do
        # ---- TYPE VERSION ----
        PT_TYPE_DIR="./save_pt_type_${SUFFIX}_${VAR}"
        LATEST_PT_TYPE=$(get_latest_epoch "${PT_TYPE_DIR}/model")

        if [ -n "$LATEST_PT_TYPE" ]; then
            echo "[TYPE ${SUFFIX} / ${VAR}] greedy eval epoch ${LATEST_PT_TYPE}..."
            # Save greedy results separately (don't overwrite topk test_result.txt)
            DIV_FLAGS=""
            [ "$USE_DIV" = "1" ] && DIV_FLAGS="-div -lamb ${LAMB} -lmd_consec ${LMD_CONSEC}"
            CUDA_VISIBLE_DEVICES= python3 main_pt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
                -m test -e ${LATEST_PT_TYPE} -b 64 \
                ${DIV_FLAGS} -t_mode greedy \
                -start_epoch ${LATEST_PT_TYPE} -epoch_step 1 \
                -i ./save_rt_type_${VAR} -o ${PT_TYPE_DIR} 2>&1 | tee "eval_greedy_type_${SUFFIX}_test_${VAR}.log" | tail -3
            # Move greedy results to separate file
            if [ -f "${PT_TYPE_DIR}/test_result.txt" ]; then
                cp "${PT_TYPE_DIR}/test_result.txt" "${PT_TYPE_DIR}/test_result_greedy.txt"
            fi
            echo "  -> ${PT_TYPE_DIR}/test_result_greedy.txt"
        else
            echo "[TYPE ${SUFFIX} / ${VAR}] No checkpoint — skip"
        fi

        # ---- NO-TYPE VERSION ----
        PT_NOTYPE_DIR="./save_pt_${SUFFIX}_${VAR}"
        LATEST_PT_NOTYPE=$(get_latest_epoch "${PT_NOTYPE_DIR}/model")

        if [ -n "$LATEST_PT_NOTYPE" ]; then
            echo "[NO-TYPE ${SUFFIX} / ${VAR}] greedy eval epoch ${LATEST_PT_NOTYPE}..."
            DIV_FLAGS=""
            [ "$USE_DIV" = "1" ] && DIV_FLAGS="-div -lamb ${LAMB} -lmd_consec ${LMD_CONSEC}"
            CUDA_VISIBLE_DEVICES= python3 main_pt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
                -m test -e ${LATEST_PT_NOTYPE} -b 64 \
                ${DIV_FLAGS} -t_mode greedy \
                -start_epoch ${LATEST_PT_NOTYPE} -epoch_step 1 \
                -i ./save_rt_kuairec_${VAR#kuairec_} -o ${PT_NOTYPE_DIR} 2>&1 | tee "eval_greedy_${SUFFIX}_test_${VAR}.log" | tail -3
            if [ -f "${PT_NOTYPE_DIR}/test_result.txt" ]; then
                cp "${PT_NOTYPE_DIR}/test_result.txt" "${PT_NOTYPE_DIR}/test_result_greedy.txt"
            fi
            echo "  -> ${PT_NOTYPE_DIR}/test_result_greedy.txt"
        else
            echo "[NO-TYPE ${SUFFIX} / ${VAR}] No checkpoint — skip"
        fi

        echo ""
    done
done

echo "############################################################"
echo "GREEDY EVALUATION COMPLETE!"
echo "############################################################"
echo ""
echo "Results saved to test_result_greedy.txt (topk results in test_result.txt)"
