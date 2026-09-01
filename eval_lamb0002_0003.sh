#!/bin/bash
# =============================================================================
# Evaluation script: λ=0.002 and λ=0.003 (type + no-type)
#
# Evaluates test set for all 4 variants:
#   Type:    save_pt_type_lamb0002_<variant> / save_pt_type_lamb0003_<variant>
#   No-type: save_pt_lamb0002_<variant>    / save_pt_lamb0003_<variant>
#
# Usage:
#   bash ~/ctrier/eval_lamb0002_0003.sh
#   nohup bash ~/ctrier/eval_lamb0002_0003.sh > ~/ctrier/eval_lamb0002_0003.log 2>&1 &
#   nohup bash ~/ctrier_type/eval_lamb0002_0003.sh > ~/ctrier_type/eval_lamb0002_0003.log 2>&1 &
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

# SUFFIX|LAMB|LMD_CONSEC
CONFIGS=(
    "lamb0002|0.002|0"
    "lamb0003|0.003|0"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

GPU=0

get_latest_epoch() {
    ls "${1}/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

echo "############################################################"
echo "# Evaluation: λ=0.002 and λ=0.003 (type + no-type)"
echo "############################################################"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX LAMB LMD_CONSEC <<< "$CONFIG"

    for VAR in "${VARIANTS[@]}"; do
        echo "=============================================="
        echo "Evaluating: ${SUFFIX} / ${VAR}"
        echo "=============================================="

        # ---- TYPE VERSION ----
        RT_TYPE_DIR="./save_rt_type_${VAR}"
        PT_TYPE_DIR="./save_pt_type_${SUFFIX}_${VAR}"
        LATEST_PT_TYPE=$(get_latest_epoch "${PT_TYPE_DIR}/model")

        if [ -n "$LATEST_PT_TYPE" ]; then
            echo "[TYPE ${SUFFIX}] Evaluating epoch ${LATEST_PT_TYPE}..."
            rm -f "${PT_TYPE_DIR}/test_result.txt"
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
                -m test -e ${LATEST_PT_TYPE} -b 64 \
                -div -lamb ${LAMB} -lmd_consec ${LMD_CONSEC} -t_mode topk \
                -start_epoch ${LATEST_PT_TYPE} -epoch_step 1 \
                -i ${RT_TYPE_DIR} -o ${PT_TYPE_DIR} 2>&1 | tee "eval_pt_type_${SUFFIX}_test_${VAR}.log"
            echo "[TYPE ${SUFFIX}] Done."
        else
            echo "[TYPE ${SUFFIX}] No checkpoint — skip"
        fi

        # ---- NO-TYPE VERSION ----
        RT_NOTYPE_DIR="./save_rt_kuairec_${VAR}"
        PT_NOTYPE_DIR="./save_pt_${SUFFIX}_${VAR}"
        LATEST_PT_NOTYPE=$(get_latest_epoch "${PT_NOTYPE_DIR}/model")

        if [ -n "$LATEST_PT_NOTYPE" ]; then
            echo "[NO-TYPE ${SUFFIX}] Evaluating epoch ${LATEST_PT_NOTYPE}..."
            rm -f "${PT_NOTYPE_DIR}/test_result.txt"
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
                -m test -e ${LATEST_PT_NOTYPE} -b 64 \
                -div -lamb ${LAMB} -lmd_consec ${LMD_CONSEC} -t_mode topk \
                -start_epoch ${LATEST_PT_NOTYPE} -epoch_step 1 \
                -i ${RT_NOTYPE_DIR} -o ${PT_NOTYPE_DIR} 2>&1 | tee "eval_pt_${SUFFIX}_test_${VAR}.log"
            echo "[NO-TYPE ${SUFFIX}] Done."
        else
            echo "[NO-TYPE ${SUFFIX}] No checkpoint — skip"
        fi

        echo ""
    done
done

echo "############################################################"
echo "EVALUATION COMPLETE!"
echo "############################################################"
echo ""
echo "Results:"
for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX _ _ <<< "$CONFIG"
    for VAR in "${VARIANTS[@]}"; do
        echo "  TYPE    ${SUFFIX} ${VAR}: ./save_pt_type_${SUFFIX}_${VAR}/test_result.txt"
        echo "  NO-TYPE ${SUFFIX} ${VAR}: ./save_pt_${SUFFIX}_${VAR}/test_result.txt"
    done
done
