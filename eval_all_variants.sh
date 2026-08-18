#!/bin/bash
# =============================================================================
# Evaluation script: runs validation and test on all trained RT and PT models.
# Evaluates ALL existing checkpoints (skips missing ones automatically).
# Generates valid_result.txt and test_result.txt in each save directory.
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
# Evaluate up to epoch 500 (the script skips missing checkpoints automatically)
MAX_EPOCHS=500

echo "=============================================="
echo "Evaluating all RT and PT models"
echo "=============================================="

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"
    PT_DIR="./save_pt_${VAR}"

    echo ""
    echo "=============================================="
    echo "Variant: ${VAR}"
    echo "=============================================="

    # ------------------------------------------------------------------
    # Step 1: Evaluate RT model (validation + test)
    # ------------------------------------------------------------------
    if [ -d "${RT_DIR}/model" ]; then
        # Find the latest checkpoint epoch
        LATEST_RT=$(ls "${RT_DIR}/model/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1)
        echo "[RT] Latest checkpoint: epoch ${LATEST_RT}"

        if [ -n "$LATEST_RT" ]; then
            # Run validation (evaluates all epochs from 1 to LATEST_RT, skipping missing)
            echo "[RT] Running validation..."
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -m valid -e ${LATEST_RT} -b 16 \
                -o ${RT_DIR} 2>&1 | tee eval_rt_valid_${VAR}.log

            # Run test
            echo "[RT] Running test..."
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -m test -e ${LATEST_RT} -b 16 \
                -o ${RT_DIR} 2>&1 | tee eval_rt_test_${VAR}.log
        else
            echo "[RT] No checkpoints found, skipping"
        fi
    else
        echo "[RT] Directory ${RT_DIR}/model not found, skipping"
    fi

    # ------------------------------------------------------------------
    # Step 2: Evaluate PT model (validation + test)
    # ------------------------------------------------------------------
    if [ -d "${PT_DIR}/model" ]; then
        # Find the latest checkpoint epoch
        LATEST_PT=$(ls "${PT_DIR}/model/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1)
        echo "[PT] Latest checkpoint: epoch ${LATEST_PT}"

        if [ -n "$LATEST_PT" ]; then
            # Run validation
            echo "[PT] Running validation..."
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -m valid -e ${LATEST_PT} -b 16 \
                -div -lamb 0.1 -t_mode topk \
                -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee eval_pt_valid_${VAR}.log

            # Run test
            echo "[PT] Running test..."
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -m test -e ${LATEST_PT} -b 16 \
                -div -lamb 0.1 -t_mode topk \
                -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee eval_pt_test_${VAR}.log
        else
            echo "[PT] No checkpoints found, skipping"
        fi
    else
        echo "[PT] Directory ${PT_DIR}/model not found, skipping"
    fi

    echo ""
    echo "=== Variant ${VAR} evaluation complete ==="
done

echo ""
echo "=============================================="
echo "ALL EVALUATIONS COMPLETE!"
echo "=============================================="
echo ""
echo "Result files generated:"
for VAR in "${VARIANTS[@]}"; do
    echo "  RT: ./save_rt_${VAR}/valid_result.txt, ./save_rt_${VAR}/test_result.txt"
    echo "  PT: ./save_pt_${VAR}/valid_result.txt, ./save_pt_${VAR}/test_result.txt"
done
