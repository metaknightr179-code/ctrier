#!/bin/bash
# =============================================================================
# Eval PT models trained with lamb=0.01 (latest checkpoint only)
# Skips variants that haven't been trained yet.
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

cd /root/ctrier

GPU=0

get_latest_epoch() {
    ls "${1}/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

echo "=============================================="
echo "Evaluating PT (lamb=0.01) models"
echo "=============================================="

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"
    PT_DIR="./save_pt_lamb001_${VAR}"

    echo ""
    echo "=============================================="
    echo "Variant: ${VAR}"
    echo "=============================================="

    # Check if PT lamb=0.01 model exists
    LATEST_PT=$(get_latest_epoch "${PT_DIR}/model")
    if [ -z "$LATEST_PT" ]; then
        echo "[PT lamb=0.01] No checkpoints found — skipping (not trained yet)"
        continue
    fi

    # Check if RT checkpoint exists (needed for PT eval)
    LATEST_RT=$(get_latest_epoch "${RT_DIR}/model")
    if [ -z "$LATEST_RT" ]; then
        echo "[RT] No checkpoints found — cannot eval PT, skipping"
        continue
    fi

    echo "[PT lamb=0.01] Evaluating epoch ${LATEST_PT}..."

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
        -div -lamb 0.01 -t_mode topk \
        -start_epoch ${LATEST_PT} -epoch_step 1 \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee eval_pt_lamb001_valid_${VAR}.log

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
        -div -lamb 0.01 -t_mode topk \
        -start_epoch ${LATEST_PT} -epoch_step 1 \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee eval_pt_lamb001_test_${VAR}.log

    echo "=== Variant ${VAR} eval complete ==="
done

echo ""
echo "=============================================="
echo "ALL lamb=0.01 EVALUATIONS COMPLETE!"
echo "=============================================="
