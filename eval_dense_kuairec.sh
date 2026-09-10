#!/bin/bash
# =============================================================================
# Evaluation for DENSE multi-position-supervision KuaiRec checkpoints.
#   save_pt_dense_<config>_<variant>         (dense, type embeddings ON)
#   save_pt_notype_dense_<config>_<variant>  (dense, -no_type)
#
# NOTE: -dense is NOT passed at eval: test_forward always gathers the last
# position; dense only changes the training loss/forward. Checkpoint weights
# are identical in structure to non-dense.
#
# Runs four protocols per checkpoint (results written into the PT dir):
#   top-k big matrix    -> test_result_topk.txt
#   top-k small matrix  -> test_result_topk_small.txt
#   greedy big matrix   -> test_result.txt         (needs save_rt_fix_<variant>)
#   greedy small matrix -> test_result_small.txt   (needs KuaiRec_small_eval/<variant>)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_dense_kuairec.sh
#   nohup bash eval_dense_kuairec.sh > eval_dense_kuairec.log 2>&1 &
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${CUDA_VISIBLE_DEVICES:-0}

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

# CONFIG: SUFFIX|LAMB|CONSEC (nodiv -> lamb 0)
CONFIGS=(
    "nodiv|0|0"
    "lamb0002|0.002|0"
    "lamb0005|0.005|0"
    "lamb0005_consec0001|0.005|0.001"
    "lamb001|0.01|0"
    "lamb005|0.05|0"
    "lamb01|0.1|0"
)

# FAMILY|DIR_PREFIX|TYPE_FLAG
FAMILIES=(
    "type|save_pt_dense_|"
    "notype|save_pt_notype_dense_|-no_type"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_denseeval_staging"
mkdir -p "$STAGE_BASE" ./rt_dummy_for_duorec

run_eval () {
    # $1=PT_DIR $2=LATEST $3=VAR_DIR $4=EF(test) $5=EN(neg) $6=TYPE_FLAG
    # $7=MODE(topk|greedy) $8=RT_DIR("" for topk) $9=OUT $10=TAG
    local PT_DIR="$1" LATEST="$2" VAR_DIR="$3" EF="$4" EN="$5" TYPE_FLAG="$6"
    local MODE="$7" RT_DIR="$8" OUT="$9" TAG="${10}"

    # Skip if up-to-date (delete the result file to force re-eval)
    local NEWER
    NEWER=$(find "${PT_DIR}/model" -name 'duorec-*.pth' -newer "$OUT" 2>/dev/null | head -1)
    if [ -s "$OUT" ] && [ -z "$NEWER" ]; then
        echo "--- ${MODE^^} [${TAG}] SKIP (up-to-date: $(basename "$OUT"))"
        return 0
    fi

    local STAGE="${STAGE_BASE}/${TAG}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE/model"

    local DIV_FLAG="-lamb 0"
    if [ "$MODE" = "greedy" ] && [ "$LAMB" != "0" ]; then
        DIV_FLAG="-div -lamb ${LAMB} -lmd_consec ${CONSEC}"
    fi
    local IN_DIR="./rt_dummy_for_duorec"
    [ "$MODE" = "greedy" ] && IN_DIR="$RT_DIR"

    echo "--- ${MODE^^} [${TAG}] $(basename "$PT_DIR") epoch ${LATEST}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf "${VAR_DIR}/train-v0.txt" \
        -vf "${VAR_DIR}/valid-v0.txt" \
        -ef "$EF" -vn "$EN" -en "$EN" \
        -cat "${VAR_DIR}/kuairec_cate.txt" \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${TYPE_FLAG} ${DIV_FLAG} -t_mode ${MODE} \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i "$IN_DIR" -o "$STAGE" 2>&1 | tail -2

    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "$OUT"
        echo "    -> $OUT"
    else
        echo "    FAILED (no test_result.txt)"
    fi
}

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX TYPE_FLAG <<< "$FAM"
    for CFG in "${CONFIGS[@]}"; do
        IFS='|' read -r SUFFIX LAMB CONSEC <<< "$CFG"
        for VAR in "${VARIANTS[@]}"; do
            PT_DIR="./${DIR_PREFIX}${SUFFIX}_${VAR}"
            [ ! -d "$PT_DIR/model" ] && { echo "SKIP: missing $PT_DIR"; continue; }
            LATEST=$(get_latest_epoch "${PT_DIR}/model")
            [ -z "$LATEST" ] && { echo "SKIP: no checkpoint in $PT_DIR"; continue; }

            VAR_DIR="./KuaiRec_variants/${VAR}"
            SMALL_DIR="./KuaiRec_small_eval/${VAR}"
            RT_DIR="./save_rt_fix_${VAR}"

            # 1. top-k, big matrix (pure accuracy, no RT)
            run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
                "${VAR_DIR}/test-v0.txt" \
                "${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                "$TYPE_FLAG" "topk" "" \
                "${PT_DIR}/test_result_topk.txt" \
                "${FAM_NAME}_${SUFFIX}_${VAR}_topk_big"

            # 2. top-k, small matrix
            if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
                run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
                    "${SMALL_DIR}/test-v0.txt" \
                    "${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                    "$TYPE_FLAG" "topk" "" \
                    "${PT_DIR}/test_result_topk_small.txt" \
                    "${FAM_NAME}_${SUFFIX}_${VAR}_topk_small"
            else
                echo "SKIP small topk: ${SMALL_DIR}/test-v0.txt missing"
            fi

            # 3. greedy, big matrix (RT beam + lambda blending)
            if [ -d "${RT_DIR}/model" ]; then
                run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
                    "${VAR_DIR}/test-v0.txt" \
                    "${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                    "$TYPE_FLAG" "greedy" "$RT_DIR" \
                    "${PT_DIR}/test_result.txt" \
                    "${FAM_NAME}_${SUFFIX}_${VAR}_greedy_big"
            else
                echo "SKIP greedy big: RT missing in ${RT_DIR}/model"
            fi

            # 4. greedy, small matrix
            if [ -f "${SMALL_DIR}/test-v0.txt" ] && [ -d "${RT_DIR}/model" ]; then
                run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
                    "${SMALL_DIR}/test-v0.txt" \
                    "${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                    "$TYPE_FLAG" "greedy" "$RT_DIR" \
                    "${PT_DIR}/test_result_small.txt" \
                    "${FAM_NAME}_${SUFFIX}_${VAR}_greedy_small"
            fi
            echo ""
        done
    done
done

echo "ALL DENSE KUAIREC EVALS DONE"
