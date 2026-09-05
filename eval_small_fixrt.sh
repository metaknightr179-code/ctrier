#!/bin/bash
# =============================================================================
# Small-matrix (canonical KuaiRec) evaluation of the fixed-RT PT checkpoints.
#
# Protocol: models were TRAINED on big-matrix-derived data (KuaiRec_variants/);
# here they are TESTED on the small-matrix leave-last-out splits built by
# convert_kuairec_to_yelp.py --interaction_file small_matrix.csv
# (output in KuaiRec_small_eval/<variant>/). Item IDs are raw KuaiRec video_ids,
# so the same 0..10727 item space and kuairec_vec.npy apply.
#
# Mirrors eval_greedy_fixrt.sh exactly (greedy inference, -b 256, same
# -n/-n_cat/-vec/-cat), only -ef/-en/-vn point at the small-matrix files.
#
# Results are written to <PT_DIR>/test_result_small.txt. The original
# test_result.txt (big-matrix results) is NOT touched: main_pt.py hardcodes its
# output filename, so the eval runs with -o pointing at a staging dir whose
# model/ is a symlink to the real checkpoint dir, and the produced
# test_result.txt is copied to test_result_small.txt afterwards.
#
# Usage: bash eval_small_fixrt.sh  (sequential, CPU, ~1-4 min per checkpoint)
# =============================================================================
cd "$(dirname "$0")"

# suffix|lamb|variant
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

STAGE_BASE="./save_smalleval_staging"
mkdir -p "$STAGE_BASE"

for T in "${TARGETS[@]}"; do
    IFS='|' read -r SUFFIX LAMB VAR <<< "$T"
    PT_DIR="./save_pt_fixrt_${SUFFIX}_${VAR}"
    RT_DIR="./save_rt_fix_${VAR}"
    SMALL_DIR="./KuaiRec_small_eval/${VAR}"

    if [ ! -d "$PT_DIR/model" ]; then echo "SKIP: missing $PT_DIR"; continue; fi
    if [ ! -d "$RT_DIR/model" ]; then echo "SKIP: missing RT $RT_DIR"; continue; fi
    if [ ! -f "${SMALL_DIR}/test-v0.txt" ]; then echo "SKIP: missing ${SMALL_DIR}/test-v0.txt"; continue; fi

    LATEST=$(get_latest_epoch "${PT_DIR}/model")
    if [ -z "$LATEST" ]; then echo "SKIP: no checkpoint in $PT_DIR"; continue; fi

    # div flag: div_loss was active in training for all lambda>0 configs
    if [ "$LAMB" == "0" ]; then DIV_FLAG=""; else DIV_FLAG="-div -lamb ${LAMB}"; fi

    # staging dir: model/ symlinked to the real checkpoint dir so the eval
    # loads the right weights but writes results outside the original dir
    STAGE="${STAGE_BASE}/fixrt_${SUFFIX}_${VAR}"
    rm -rf "$STAGE"
    mkdir -p "$STAGE"
    ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE/model"

    echo "=============================================="
    echo "SMALL-matrix eval: fixrt ${SUFFIX} / ${VAR} (epoch ${LATEST}, lamb=${LAMB})"
    echo "=============================================="

    python3 main_pt.py \
        -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
        -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
        -ef ${SMALL_DIR}/test-v0.txt \
        -vn ${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${DIV_FLAG} -lmd_consec 0 -t_mode greedy \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i ${RT_DIR} -o ${STAGE} 2>&1 | tee "eval_small_fixrt_${SUFFIX}_${VAR}.log"

    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "${PT_DIR}/test_result_small.txt"
        echo "=== ${SUFFIX}/${VAR} done -> ${PT_DIR}/test_result_small.txt ==="
    else
        echo "=== ${SUFFIX}/${VAR} FAILED (no test_result.txt produced) ==="
    fi
    echo ""
done

echo "ALL SMALL-MATRIX FIXRT EVALS DONE"
