#!/bin/bash
# Evaluate the 7 newly transferred type-embedding checkpoints (test mode only)
# Runs locally on CPU (slow but works)

cd /Users/notrobin/Documents/trae_projects/trier

# config|lamb|lmd_consec|variant  (only configs with epoch 500 ckpt + no test_result yet)
TARGETS=(
    "nodiv|0|0|kuairec_first_individual"
    "nodiv|0|0|kuairec_first_average"
    "lamb0005|0.005|0|kuairec_highest_individual"
    "lamb0005|0.005|0|kuairec_highest_average"
    "lamb0005|0.005|0|kuairec_first_individual"
    "lamb0005|0.005|0|kuairec_first_average"
    "lamb001|0.01|0|kuairec_highest_individual"
)

for T in "${TARGETS[@]}"; do
    IFS='|' read -r SUFFIX LAMB LMD_CONSEC VAR <<< "$T"
    PT_DIR="./save_pt_type_${SUFFIX}_${VAR}"
    RT_DIR="./save_rt_type_${VAR}"

    echo "=============================================="
    echo "Evaluating: ${SUFFIX} / ${VAR}  (lamb=${LAMB})"
    echo "=============================================="

    if [ ! -f "${PT_DIR}/model/duorec-500.pth" ]; then
        echo "SKIP: no epoch 500 checkpoint"
        continue
    fi

    # Build div_flag: pass -div for all (topk mode ignores lamb, so safe;
    # keeps consistent with eval_type.sh)
    DIV_FLAG="-div -lamb ${LAMB}"
    if [ "$LAMB" == "0" ]; then
        DIV_FLAG=""  # match training: nodiv was trained without -div
    fi

    rm -f "${PT_DIR}/test_result.txt"
    python3 main_pt.py \
        -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
        -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
        -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
        -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e 500 -b 64 \
        ${DIV_FLAG} -lmd_consec ${LMD_CONSEC} -t_mode topk \
        -start_epoch 500 -epoch_step 1 \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_pt_type_${SUFFIX}_test_${VAR}.log"

    echo "=== ${SUFFIX}/${VAR} done ==="
    echo ""
done

echo "All evals complete. Results:"
for T in "${TARGETS[@]}"; do
    IFS='|' read -r SUFFIX _ _ VAR <<< "$T"
    echo "  ${SUFFIX}/${VAR}: ./save_pt_type_${SUFFIX}_${VAR}/test_result.txt"
done
