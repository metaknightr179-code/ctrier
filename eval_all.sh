#!/bin/bash
# =============================================================================
# Unified evaluation script: evaluates ALL PT variants and baselines.
#   - nodiv, lamb0005, lamb001, lamb005, lamb01, consec0001
#   - GRU4Rec, SASRec, BERT4Rec (eval-only)
# for all 4 Kuairec variants.
#
# Uses fixed NDCG formula (1/log2(idx+2)).
#
# Usage:
#   bash eval_all.sh
#   nohup bash eval_all.sh > eval_all.log 2>&1 &
# =============================================================================

VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_highest_average"
    "kuairec_first_individual"
    "kuairec_first_average"
)

# PT configs: SUFFIX|LAMB|LMD_CONSEC
PT_CONFIGS=(
    "nodiv|0|0"
    "lamb0005|0.005|0"
    "lamb001|0.01|0"
    "lamb005|0.05|0"
    "lamb01|0.1|0"
    "consec0001|0.01|0.001"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

GPU=0
ITEM_NUM=10728
N_CAT=31
MAXLEN=50

get_latest_epoch() {
    ls "${1}/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

echo "############################################################"
echo "# Unified Evaluation: All PT Variants + Baselines"
echo "############################################################"
echo ""

# ===================== PT MODELS =====================
for CONFIG in "${PT_CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX LAMB LMD_CONSEC <<< "$CONFIG"

    echo "############################################################"
    echo "# PT Config: ${SUFFIX} (lamb=${LAMB}, lmd_consec=${LMD_CONSEC})"
    echo "############################################################"

    for VAR in "${VARIANTS[@]}"; do
        RT_DIR="./save_rt_${VAR}"
        PT_DIR="./save_pt_${SUFFIX}_${VAR}"

        echo ""
        echo "=============================================="
        echo "PT ${SUFFIX} — ${VAR}"
        echo "=============================================="

        LATEST_PT=$(get_latest_epoch "${PT_DIR}/model")
        if [ -z "$LATEST_PT" ]; then
            echo "[PT ${SUFFIX}] No checkpoints — skipping"
            continue
        fi

        LATEST_RT=$(get_latest_epoch "${RT_DIR}/model")
        if [ -z "$LATEST_RT" ]; then
            echo "[RT] No checkpoints — cannot eval, skipping"
            continue
        fi

        echo "[PT ${SUFFIX}] Evaluating epoch ${LATEST_PT}..."

        # Validation
        rm -f "${PT_DIR}/valid_result.txt"
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -m valid -e ${LATEST_PT} -b 16 \
            -div -lamb ${LAMB} -lmd_consec ${LMD_CONSEC} -t_mode topk \
            -start_epoch ${LATEST_PT} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_pt_${SUFFIX}_valid_${VAR}.log"

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
            -div -lamb ${LAMB} -lmd_consec ${LMD_CONSEC} -t_mode topk \
            -start_epoch ${LATEST_PT} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_pt_${SUFFIX}_test_${VAR}.log"

        echo "=== ${SUFFIX}/${VAR} eval complete ==="
    done
    echo ""
done

# ===================== BASELINES =====================
echo "############################################################"
echo "# Baselines (eval-only with fixed NDCG)"
echo "############################################################"
echo ""

for VAR in "${VARIANTS[@]}"; do
    DATA_DIR="./KuaiRec_variants/${VAR}"
    OUTPUT_DIR="./baseline_results_${VAR}"
    mkdir -p "${OUTPUT_DIR}"

    echo "=============================================="
    echo "Baselines — ${VAR}"
    echo "=============================================="

    # GRU4Rec
    GRU_DIR="./save_gru4rec_${VAR}"
    if [ -f "${GRU_DIR}/gru4rec_best.pth" ]; then
        echo "[GRU4Rec] Evaluating..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 gru4rec_pytorch.py \
            --eval_only --ckpt_dir "${GRU_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --batch_size 256 \
            --maxlen ${MAXLEN} \
            --cat "${DATA_DIR}/kuairec_cate.txt" \
            --n_cat ${N_CAT} \
            --vec "./KuaiRec_variants/kuairec_vec.npy" \
            --output "${OUTPUT_DIR}/gru4rec_results.txt" 2>&1 | tee "eval_gru4rec_${VAR}.log"
    else
        echo "[GRU4Rec] No checkpoint — skip"
    fi

    # SASRec
    SAS_DIR="./save_sasrec_${VAR}"
    if [ -f "${SAS_DIR}/sasrec_best.pth" ]; then
        echo "[SASRec] Evaluating..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 sasrec_pytorch.py \
            --eval_only --ckpt_dir "${SAS_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUTPUT_DIR}/sasrec_results.txt" 2>&1 | tee "eval_sasrec_${VAR}.log"
    elif [ -f "./sasrec_best.pth" ]; then
        echo "[SASRec] Using shared checkpoint..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 sasrec_pytorch.py \
            --eval_only --ckpt_path "sasrec_best.pth" --ckpt_dir "." \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUTPUT_DIR}/sasrec_results.txt" 2>&1 | tee "eval_sasrec_${VAR}.log"
    else
        echo "[SASRec] No checkpoint — skip"
    fi

    # BERT4Rec
    BERT_DIR="./save_bert4rec_${VAR}"
    if [ -f "${BERT_DIR}/bert4rec_best.pth" ]; then
        echo "[BERT4Rec] Evaluating..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 bert4rec_pytorch.py \
            --eval_only --ckpt_dir "${BERT_DIR}" \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUTPUT_DIR}/bert4rec_results.txt" 2>&1 | tee "eval_bert4rec_${VAR}.log"
    elif [ -f "./bert4rec_best.pth" ]; then
        echo "[BERT4Rec] Using shared checkpoint..."
        CUDA_VISIBLE_DEVICES=${GPU} python3 bert4rec_pytorch.py \
            --eval_only --ckpt_path "bert4rec_best.pth" --ckpt_dir "." \
            --test_file "${DATA_DIR}/test-v0.txt" \
            --item_num ${ITEM_NUM} \
            --maxlen ${MAXLEN} \
            --output "${OUTPUT_DIR}/bert4rec_results.txt" 2>&1 | tee "eval_bert4rec_${VAR}.log"
    else
        echo "[BERT4Rec] No checkpoint — skip"
    fi

    echo "=== ${VAR} baselines complete ==="
    echo ""
done

echo "############################################################"
echo "ALL EVALUATIONS COMPLETE!"
echo "############################################################"
echo ""
echo "Results:"
for CONFIG in "${PT_CONFIGS[@]}"; do
    IFS='|' read -r SUFFIX _ _ <<< "$CONFIG"
    for VAR in "${VARIANTS[@]}"; do
        echo "  PT ${SUFFIX} ${VAR}: ./save_pt_${SUFFIX}_${VAR}/test_result.txt"
    done
done
for VAR in "${VARIANTS[@]}"; do
    echo "  Baselines ${VAR}: ./baseline_results_${VAR}/"
done