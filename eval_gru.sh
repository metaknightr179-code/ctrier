#!/bin/bash
# =============================================================================
# GRU Evaluation Script
# Evaluates all GRU-PT models (valid + test, topk mode)
#
# Usage:
#   bash eval_gru.sh
#   nohup bash eval_gru.sh > eval_gru.log 2>&1 &
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

PT_CONFIGS=(
    "nodiv|0|0"
    "lamb0005|0.005|0"
    "lamb001|0.01|0"
    "lamb005|0.05|0"
    "lamb01|0.1|0"
    "consec0001|0.01|0.01"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

GPU=0
ITEM_NUM=10728
N_CAT=31
MAXLEN=50

get_latest_epoch() {
    ls "${1}/"duorec-*.pth 2>/dev/null | grep -v tmp | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

echo "############################################################"
echo "# GRU Evaluation: All PT Variants"
echo "############################################################"
echo ""

for CONFIG in "${PT_CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX LAMB LMD_CONSEC <<< "$CONFIG"

    echo "############################################################"
    echo "# GRU-PT Config: ${SUFFIX} (lamb=${LAMB}, lmd_consec=${LMD_CONSEC})"
    echo "############################################################"

    for VAR in "${VARIANTS[@]}"; do
        RT_DIR="./save_rt_gru_${VAR}"
        PT_DIR="./save_pt_gru_${SUFFIX}_${VAR}"

        echo ""
        echo "=============================================="
        echo "GRU-PT ${SUFFIX} — ${VAR}"
        echo "=============================================="

        LATEST_PT=$(get_latest_epoch "${PT_DIR}/model")
        if [ -z "$LATEST_PT" ]; then
            echo "[GRU-PT ${SUFFIX}] No checkpoints — skipping"
            continue
        fi

        LATEST_RT=$(get_latest_epoch "${RT_DIR}/model")
        if [ -z "$LATEST_RT" ]; then
            echo "[GRU-RT] No checkpoints — cannot eval, skipping"
            continue
        fi

        echo "[GRU-PT ${SUFFIX}] Evaluating epoch ${LATEST_PT}..."

        DIV_FLAG="-div -lamb ${LAMB}"
        if [ "$LAMB" == "0" ]; then
            DIV_FLAG=""
        fi

        # Validation
        rm -f "${PT_DIR}/valid_result.txt"
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_gru.py \
            -model_type pt \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
            -m valid -e ${LATEST_PT} -b 64 \
            ${DIV_FLAG} -lmd_consec ${LMD_CONSEC} -t_mode topk \
            -start_epoch ${LATEST_PT} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_gru_${SUFFIX}_valid_${VAR}.log"

        # Test
        rm -f "${PT_DIR}/test_result.txt"
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_gru.py \
            -model_type pt \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
            -m test -e ${LATEST_PT} -b 64 \
            ${DIV_FLAG} -lmd_consec ${LMD_CONSEC} -t_mode topk \
            -start_epoch ${LATEST_PT} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_gru_${SUFFIX}_test_${VAR}.log"

        echo "=== GRU ${SUFFIX}/${VAR} eval complete ==="
    done
    echo ""
done

echo "############################################################"
echo "GRU EVALUATIONS COMPLETE!"
echo "############################################################"
echo ""
echo "Results:"
for CONFIG in "${PT_CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX _ _ <<< "$CONFIG"
    for VAR in "${VARIANTS[@]}"; do
        echo "  GRU-PT ${SUFFIX} ${VAR}: ./save_pt_gru_${SUFFIX}_${VAR}/test_result.txt"
    done
done
