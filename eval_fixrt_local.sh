#!/bin/bash
# =============================================================================
# Local (CPU) test-mode evaluation of the fixed-RT PT checkpoints.
# Evaluates: save_pt_fixrt_<config>_<variant>  (RT: save_rt_fix_<variant>)
# Configs present locally: nodiv (4), lamb0002 (4), lamb0005 (2)
# =============================================================================
cd /Users/notrobin/Documents/trae_projects/trier

# config|lamb|variant
TARGETS=(
    "nodiv|0|kuairec_highest_individual"
    "nodiv|0|kuairec_highest_average"
    "nodiv|0|kuairec_first_individual"
    "nodiv|0|kuairec_first_average"
    "lamb0002|0.002|kuairec_highest_individual"
    "lamb0002|0.002|kuairec_highest_average"
    "lamb0002|0.002|kuairec_first_individual"
    "lamb0002|0.002|kuairec_first_average"
    "lamb0005|0.005|kuairec_highest_individual"
    "lamb0005|0.005|kuairec_highest_average"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

for T in "${TARGETS[@]}"; do
    IFS='|' read -r SUFFIX LAMB VAR <<< "$T"
    PT_DIR="./save_pt_fixrt_${SUFFIX}_${VAR}"
    RT_DIR="./save_rt_fix_${VAR}"

    if [ ! -d "$PT_DIR" ]; then echo "SKIP: missing $PT_DIR"; continue; fi
    if [ ! -d "$RT_DIR/model" ]; then echo "SKIP: missing RT $RT_DIR"; continue; fi

    LATEST=$(get_latest_epoch "${PT_DIR}/model")
    if [ -z "$LATEST" ]; then echo "SKIP: no checkpoint in $PT_DIR"; continue; fi

    if [ "$LAMB" == "0" ]; then DIV_FLAG=""; else DIV_FLAG="-div -lamb ${LAMB}"; fi

    echo "=============================================="
    echo "Evaluating: ${SUFFIX} / ${VAR} (epoch ${LATEST}, lamb=${LAMB})"
    echo "=============================================="

    rm -f "${PT_DIR}/test_result.txt"
    python3 main_pt.py \
        -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
        -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
        -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
        -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 64 \
        ${DIV_FLAG} -lmd_consec 0 -t_mode topk \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_pt_fixrt_${SUFFIX}_test_${VAR}.log"

    echo "=== ${SUFFIX}/${VAR} done ==="
    echo ""
done

echo "ALL FIXRT EVALS DONE"
