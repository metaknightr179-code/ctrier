#!/bin/bash
# =============================================================================
# Evaluate PT models trained with nodiv (lamb=0, lmd_consec=0)
# Uses fixed NDCG formula from updated script.py
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

get_latest_epoch() {
    ls "${1}/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

echo "=============================================="
echo "Evaluating PT (nodiv: lamb=0, lmd_consec=0) models"
echo "=============================================="

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"
    PT_DIR="./save_pt_nodiv_${VAR}"

    echo ""
    echo "=============================================="
    echo "Variant: ${VAR}"
    echo "=============================================="

    # Check if PT nodiv model exists
    LATEST_PT=$(get_latest_epoch "${PT_DIR}/model")
    if [ -z "$LATEST_PT" ]; then
        echo "[PT nodiv] No checkpoints found — skipping (not trained yet)"
        continue
    fi

    # Check if RT checkpoint exists (needed for PT eval)
    LATEST_RT=$(get_latest_epoch "${RT_DIR}/model")
    if [ -z "$LATEST_RT" ]; then
        echo "[RT] No checkpoints found — cannot eval PT, skipping"
        continue
    fi

    echo "[PT nodiv] Evaluating epoch ${LATEST_PT}..."

    # Validate
    rm -f "${PT_DIR}/valid_result.txt"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
        -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
        -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
        -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
        -n 10728 -m valid -e ${LATEST_PT} -b 16 \
        -div -lamb 0 -lmd_consec 0 -t_mode topk \
        -start_epoch ${LATEST_PT} -epoch_step 1 \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_pt_nodiv_valid_${VAR}.log"

    # Test
    rm -f "${PT_DIR}/test_result.txt"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
        -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
        -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
        -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
        -n 10728 -m test -e ${LATEST_PT} -b 16 \
        -div -lamb 0 -lmd_consec 0 -t_mode topk \
        -start_epoch ${LATEST_PT} -epoch_step 1 \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_pt_nodiv_test_${VAR}.log"

    echo "=== Variant ${VAR} eval complete ==="
done

echo ""
echo "=============================================="
echo "ALL NODIV EVALUATIONS COMPLETE!"
echo "=============================================="