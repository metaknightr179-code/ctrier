#!/bin/bash
# =============================================================================
# Smart evaluation script:
#   Phase 1: Validate every 10th epoch (fast — ~50 evals instead of 500)
#   Phase 2: Find best validation epoch from Phase 1 results
#   Phase 3: Test ONLY the best epoch (1 eval)
# Uses tmux + nohup to survive SSH disconnects.
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

cd /root/ctrier

GPU=0
EPOCH_STEP=10  # Validate every 10th epoch

echo "=============================================="
echo "Smart Evaluation: best epochs only"
echo "Step: every ${EPOCH_STEP}th epoch for validation"
echo "=============================================="

for VAR in "${VARIANTS[@]}"; do
    RT_DIR="./save_rt_${VAR}"
    PT_DIR="./save_pt_${VAR}"

    echo ""
    echo "=============================================="
    echo "Variant: ${VAR}"
    echo "=============================================="

    # ------------------------------------------------------------------
    # Helper: find latest checkpoint epoch in a model directory
    # ------------------------------------------------------------------
    get_latest_epoch() {
        local model_dir="$1"
        ls "${model_dir}/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
    }

    # Helper: count checkpoints in a model directory
    count_checkpoints() {
        local model_dir="$1"
        ls "${model_dir}/"duorec-*.pth 2>/dev/null | wc -l
    }

    # Helper: find best validation epoch from valid_result.txt
    # Reads Python dict lines, finds epoch with highest recall@5_f (fallback recall@5)
    # Returns 0 if file is missing/empty (caller should fall back to latest)
    get_best_val_epoch() {
        local valid_file="$1"
        if [ ! -f "$valid_file" ] || [ ! -s "$valid_file" ]; then
            echo "0"
            return
        fi
        python3 -c "
import ast, sys, re
best_ep, best_r5 = 0, -1
with open('$valid_file') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('{'): continue
        try:
            clean = re.sub(r'np\.float\d+\(([^)]+)\)', r'\1', line)
            d = ast.literal_eval(clean)
        except Exception:
            continue
        r5 = d.get('recall@5_f', d.get('recall@5', -1))
        if isinstance(r5, (int, float)) and r5 > best_r5:
            best_r5 = r5
            best_ep = d.get('epoch', 0)
print(best_ep)
"
    }

    # ------------------------------------------------------------------
    # Step 1: Evaluate RT model
    # ------------------------------------------------------------------
    LATEST_RT=$(get_latest_epoch "${RT_DIR}/model")
    NUM_RT_CKPTS=$(count_checkpoints "${RT_DIR}/model")
    if [ -n "$LATEST_RT" ]; then
        echo "[RT] Latest checkpoint: epoch ${LATEST_RT} (${NUM_RT_CKPTS} total)"

        # Phase 1: Validate every 10th epoch — UNLESS only 1 checkpoint exists
        if [ "$NUM_RT_CKPTS" -le 1 ]; then
            echo "[RT] Phase 1: Only 1 checkpoint, validating just epoch ${LATEST_RT}..."
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
        else
            echo "[RT] Phase 1: Validating every ${EPOCH_STEP}th epoch (1 to ${LATEST_RT})..."
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -m valid -e ${LATEST_RT} -b 16 \
                -start_epoch 1 -epoch_step ${EPOCH_STEP} \
                -o ${RT_DIR} 2>&1 | tee eval_rt_valid_${VAR}.log
        fi

        # Phase 2: Find best validation epoch (fallback to latest if no results)
        BEST_RT_EPOCH=$(get_best_val_epoch "${RT_DIR}/valid_result.txt")
        if [ -z "$BEST_RT_EPOCH" ] || [ "$BEST_RT_EPOCH" = "0" ]; then
            echo "[RT] Phase 2: No validation results, falling back to latest epoch ${LATEST_RT}"
            BEST_RT_EPOCH=$LATEST_RT
        else
            echo "[RT] Phase 2: Best validation epoch: ${BEST_RT_EPOCH}"
        fi

        # Phase 3: Test only the best epoch
        echo "[RT] Phase 3: Testing best epoch ${BEST_RT_EPOCH}..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -m test -e ${BEST_RT_EPOCH} -b 16 \
            -start_epoch ${BEST_RT_EPOCH} -epoch_step 1 \
            -o ${RT_DIR} 2>&1 | tee eval_rt_test_${VAR}.log
    else
        echo "[RT] No checkpoints found, skipping"
    fi

    # ------------------------------------------------------------------
    # Step 2: Evaluate PT model
    # ------------------------------------------------------------------
    LATEST_PT=$(get_latest_epoch "${PT_DIR}/model")
    NUM_PT_CKPTS=$(count_checkpoints "${PT_DIR}/model")
    if [ -n "$LATEST_PT" ]; then
        echo "[PT] Latest checkpoint: epoch ${LATEST_PT} (${NUM_PT_CKPTS} total)"

        # Phase 1: Validate every 10th epoch — UNLESS only 1 checkpoint exists
        if [ "$NUM_PT_CKPTS" -le 1 ]; then
            echo "[PT] Phase 1: Only 1 checkpoint, validating just epoch ${LATEST_PT}..."
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
        else
            echo "[PT] Phase 1: Validating every ${EPOCH_STEP}th epoch (1 to ${LATEST_PT})..."
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -m valid -e ${LATEST_PT} -b 16 \
                -div -lamb 0.1 -t_mode topk \
                -start_epoch 1 -epoch_step ${EPOCH_STEP} \
                -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee eval_pt_valid_${VAR}.log
        fi

        # Phase 2: Find best validation epoch (fallback to latest if no results)
        BEST_PT_EPOCH=$(get_best_val_epoch "${PT_DIR}/valid_result.txt")
        if [ -z "$BEST_PT_EPOCH" ] || [ "$BEST_PT_EPOCH" = "0" ]; then
            echo "[PT] Phase 2: No validation results, falling back to latest epoch ${LATEST_PT}"
            BEST_PT_EPOCH=$LATEST_PT
        else
            echo "[PT] Phase 2: Best validation epoch: ${BEST_PT_EPOCH}"
        fi

        # Phase 3: Test only the best epoch
        echo "[PT] Phase 3: Testing best epoch ${BEST_PT_EPOCH}..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -m test -e ${BEST_PT_EPOCH} -b 16 \
            -div -lamb 0.1 -t_mode topk \
            -start_epoch ${BEST_PT_EPOCH} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee eval_pt_test_${VAR}.log
    else
        echo "[PT] No checkpoints found, skipping"
    fi

    # ------------------------------------------------------------------
    # Step 3: Evaluate PT-no-consec model (same as PT but with -no_consec flag)
    # ------------------------------------------------------------------
    PT_NC_DIR="./save_pt_no_consec_${VAR}"
    LATEST_PT_NC=$(get_latest_epoch "${PT_NC_DIR}/model")
    NUM_PT_NC_CKPTS=$(count_checkpoints "${PT_NC_DIR}/model")
    if [ -n "$LATEST_PT_NC" ]; then
        echo "[PT-NC] Latest checkpoint: epoch ${LATEST_PT_NC} (${NUM_PT_NC_CKPTS} total)"

        # Phase 1: Validate
        if [ "$NUM_PT_NC_CKPTS" -le 1 ]; then
            echo "[PT-NC] Phase 1: Only 1 checkpoint, validating just epoch ${LATEST_PT_NC}..."
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
        else
            echo "[PT-NC] Phase 1: Validating every ${EPOCH_STEP}th epoch (1 to ${LATEST_PT_NC})..."
            CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
                -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
                -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
                -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
                -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
                -n 10728 -m valid -e ${LATEST_PT_NC} -b 16 \
                -div -lamb 0.1 -t_mode topk -no_consec \
                -start_epoch 1 -epoch_step ${EPOCH_STEP} \
                -i ${RT_DIR} -o ${PT_NC_DIR} 2>&1 | tee eval_pt_nc_valid_${VAR}.log
        fi

        # Phase 2
        BEST_PT_NC_EPOCH=$(get_best_val_epoch "${PT_NC_DIR}/valid_result.txt")
        if [ -z "$BEST_PT_NC_EPOCH" ] || [ "$BEST_PT_NC_EPOCH" = "0" ]; then
            echo "[PT-NC] Phase 2: No validation results, falling back to latest epoch ${LATEST_PT_NC}"
            BEST_PT_NC_EPOCH=$LATEST_PT_NC
        else
            echo "[PT-NC] Phase 2: Best validation epoch: ${BEST_PT_NC_EPOCH}"
        fi

        # Phase 3: Test
        echo "[PT-NC] Phase 3: Testing best epoch ${BEST_PT_NC_EPOCH}..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -m test -e ${BEST_PT_NC_EPOCH} -b 16 \
            -div -lamb 0.1 -t_mode topk -no_consec \
            -start_epoch ${BEST_PT_NC_EPOCH} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_NC_DIR} 2>&1 | tee eval_pt_nc_test_${VAR}.log
    else
        echo "[PT-NC] No checkpoints found, skipping"
    fi

    echo ""
    echo "=== Variant ${VAR} evaluation complete ==="
    echo "  RT best epoch:      ${BEST_RT_EPOCH:-N/A}"
    echo "  PT best epoch:      ${BEST_PT_EPOCH:-N/A}"
    echo "  PT-NC best epoch:   ${BEST_PT_NC_EPOCH:-N/A}"
done

echo ""
echo "=============================================="
echo "ALL EVALUATIONS COMPLETE!"
echo "=============================================="
echo ""
echo "Result files:"
for VAR in "${VARIANTS[@]}"; do
    echo "  RT:    ./save_rt_${VAR}/valid_result.txt, ./save_rt_${VAR}/test_result.txt"
    echo "  PT:    ./save_pt_${VAR}/valid_result.txt, ./save_pt_${VAR}/test_result.txt"
    echo "  PT-NC: ./save_pt_no_consec_${VAR}/valid_result.txt, ./save_pt_no_consec_${VAR}/test_result.txt"
done
