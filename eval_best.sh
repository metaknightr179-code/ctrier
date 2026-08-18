#!/bin/bash
# =============================================================================
# Simple evaluation script: validate + test ONLY the latest checkpoint (epoch 500)
# Skips Phase 1 (every 10th epoch) and Phase 2 (find best) — just evals the latest.
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

cd /root/ctrier

GPU=0

echo "=============================================="
echo "Simple Evaluation: latest checkpoint only"
echo "=============================================="

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"
    PT_DIR="./save_pt_${VAR}"
    PT_NC_DIR="./save_pt_no_consec_${VAR}"

    echo ""
    echo "=============================================="
    echo "Variant: ${VAR}"
    echo "=============================================="

    # Helper: find latest checkpoint epoch
    get_latest_epoch() {
        ls "${1}/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
    }

    # ------------------------------------------------------------------
    # RT model
    # ------------------------------------------------------------------
    LATEST_RT=$(get_latest_epoch "${RT_DIR}/model")
    if [ -n "$LATEST_RT" ]; then
        echo "[RT] Evaluating epoch ${LATEST_RT}..."

        # Validate
        rm -f "${RT_DIR}/valid_result.txt"
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -m valid -e ${LATEST_RT} -b 16 \
            -start_epoch ${LATEST_RT} -epoch_step 1 \
            -o ${RT_DIR} 2>&1 | tee eval_rt_valid_${VAR}.log

        # Test
        rm -f "${RT_DIR}/test_result.txt"
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -m test -e ${LATEST_RT} -b 16 \
            -start_epoch ${LATEST_RT} -epoch_step 1 \
            -o ${RT_DIR} 2>&1 | tee eval_rt_test_${VAR}.log
    else
        echo "[RT] No checkpoints found, skipping"
    fi

    # ------------------------------------------------------------------
    # PT model (with consec loss)
    # ------------------------------------------------------------------
    LATEST_PT=$(get_latest_epoch "${PT_DIR}/model")
    if [ -n "$LATEST_PT" ]; then
        echo "[PT] Evaluating epoch ${LATEST_PT}..."

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
            -div -lamb 0.1 -t_mode topk \
            -start_epoch ${LATEST_PT} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee eval_pt_valid_${VAR}.log

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
            -div -lamb 0.1 -t_mode topk \
            -start_epoch ${LATEST_PT} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee eval_pt_test_${VAR}.log
    else
        echo "[PT] No checkpoints found, skipping"
    fi

    # ------------------------------------------------------------------
    # PT-no-consec model
    # ------------------------------------------------------------------
    LATEST_PT_NC=$(get_latest_epoch "${PT_NC_DIR}/model")
    if [ -n "$LATEST_PT_NC" ]; then
        echo "[PT-NC] Evaluating epoch ${LATEST_PT_NC}..."

        # Validate
        rm -f "${PT_NC_DIR}/valid_result.txt"
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -m valid -e ${LATEST_PT_NC} -b 16 \
            -div -lamb 0.1 -t_mode topk -no_consec \
            -start_epoch ${LATEST_PT_NC} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_NC_DIR} 2>&1 | tee eval_pt_nc_valid_${VAR}.log

        # Test
        rm -f "${PT_NC_DIR}/test_result.txt"
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -m test -e ${LATEST_PT_NC} -b 16 \
            -div -lamb 0.1 -t_mode topk -no_consec \
            -start_epoch ${LATEST_PT_NC} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_NC_DIR} 2>&1 | tee eval_pt_nc_test_${VAR}.log
    else
        echo "[PT-NC] No checkpoints found, skipping"
    fi

    echo ""
    echo "=== Variant ${VAR} evaluation complete ==="
done

echo ""
echo "=============================================="
echo "ALL EVALUATIONS COMPLETE!"
echo "=============================================="
