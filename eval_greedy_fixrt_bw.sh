#!/bin/bash
# =============================================================================
# GREEDY-mode evaluation with LARGER BEAM parameters on BOTH big and small matrices.
#
# Default TRIER beam (bw=3, k=5) generates only 15 candidates per session —
# 0.14% of KuaiRec's 10,728-item catalog. This script tests larger beams to
# see if the greedy mode accuracy bottleneck is candidate pool size.
#
# Tests two beam configs:
#   bw=10, k=10  -> 100 candidates (0.93% of catalog)
#   bw=20, k=20  -> 400 candidates (3.7% of catalog)
#
# Only evaluates the nodiv config (pure accuracy, no diversity blending) for
# both type and notype families, on all 4 variants.
#
# Output (per PT checkpoint dir):
#   Big matrix:   test_result_greedy_bw10_k10.txt,  test_result_greedy_bw20_k20.txt
#   Small matrix: test_result_greedy_small_bw10_k10.txt,  test_result_greedy_small_bw20_k20.txt
#
# Usage: nohup bash eval_greedy_fixrt_bw.sh > eval_greedy_bw.log 2>&1 &
# =============================================================================
cd "$(dirname "$0")"

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

# FAMILY|DIR_PREFIX|TYPE_FLAG
FAMILIES=(
    "type|save_pt_fixrt_|"
    "notype|save_pt_notype_fixrt_|-no_type"
)

# BW|K|LABEL
BEAM_CONFIGS=(
    "10|10|bw10_k10"
    "20|20|bw20_k20"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_greedy_bw_staging"
mkdir -p "$STAGE_BASE"

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX TYPE_FLAG <<< "$FAM"
    for BC in "${BEAM_CONFIGS[@]}"; do
        IFS='|' read -r BW K BEAM_LABEL <<< "$BC"
        for VAR in "${VARIANTS[@]}"; do
            PT_DIR="./${DIR_PREFIX}nodiv_${VAR}"
            RT_DIR="./save_rt_fix_${VAR}"
            VAR_DIR="./KuaiRec_variants/${VAR}"
            SMALL_DIR="./KuaiRec_small_eval/${VAR}"

            [ ! -d "$PT_DIR/model" ] && { echo "SKIP: missing $PT_DIR"; continue; }
            [ ! -d "$RT_DIR/model" ] && { echo "SKIP: missing RT $RT_DIR"; continue; }
            LATEST=$(get_latest_epoch "${PT_DIR}/model")
            [ -z "$LATEST" ] && { echo "SKIP: no checkpoint in $PT_DIR"; continue; }

            # ---- BIG MATRIX ----
            TAG="${FAM_NAME}_nodiv_${VAR}_big_${BEAM_LABEL}"
            STAGE="${STAGE_BASE}/${TAG}"
            rm -rf "$STAGE"; mkdir -p "$STAGE"
            ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE/model"

            echo "=============================================="
            echo "GREEDY big-matrix [${TAG}] epoch ${LATEST} (bw=${BW}, k=${K})"
            echo "=============================================="
            python3 main_pt.py \
                -tf "${VAR_DIR}/train-v0.txt" \
                -vf "${VAR_DIR}/valid-v0.txt" \
                -ef "${VAR_DIR}/test-v0.txt" \
                -vn "${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                -en "${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                -cat "${VAR_DIR}/kuairec_cate.txt" \
                -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
                -m test -e ${LATEST} -b 256 \
                ${TYPE_FLAG} -lamb 0 -t_mode greedy \
                -bw ${BW} -k ${K} \
                -start_epoch ${LATEST} -epoch_step 1 \
                -i "$RT_DIR" -o "$STAGE" 2>&1 | tail -5

            RESULT_FILE="test_result_greedy_${BEAM_LABEL}.txt"
            if [ -f "${STAGE}/test_result.txt" ]; then
                cp "${STAGE}/test_result.txt" "${PT_DIR}/${RESULT_FILE}"
                echo "    -> ${PT_DIR}/${RESULT_FILE}"
            else
                echo "    FAILED (no test_result.txt)"
            fi

            # ---- SMALL MATRIX ----
            if [ -d "$SMALL_DIR" ]; then
                TAG_S="${FAM_NAME}_nodiv_${VAR}_small_${BEAM_LABEL}"
                STAGE_S="${STAGE_BASE}/${TAG_S}"
                rm -rf "$STAGE_S"; mkdir -p "$STAGE_S"
                ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE_S/model"

                echo "=============================================="
                echo "GREEDY small-matrix [${TAG_S}] epoch ${LATEST} (bw=${BW}, k=${K})"
                echo "=============================================="
                python3 main_pt.py \
                    -tf "${SMALL_DIR}/train-v0.txt" \
                    -vf "${SMALL_DIR}/valid-v0.txt" \
                    -ef "${SMALL_DIR}/test-v0.txt" \
                    -vn "${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                    -en "${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                    -cat "${SMALL_DIR}/kuairec_cate.txt" \
                    -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
                    -m test -e ${LATEST} -b 256 \
                    ${TYPE_FLAG} -lamb 0 -t_mode greedy \
                    -bw ${BW} -k ${K} \
                    -start_epoch ${LATEST} -epoch_step 1 \
                    -i "$RT_DIR" -o "$STAGE_S" 2>&1 | tail -5

                RESULT_FILE_S="test_result_greedy_small_${BEAM_LABEL}.txt"
                if [ -f "${STAGE_S}/test_result_small.txt" ]; then
                    cp "${STAGE_S}/test_result_small.txt" "${PT_DIR}/${RESULT_FILE_S}"
                    echo "    -> ${PT_DIR}/${RESULT_FILE_S}"
                elif [ -f "${STAGE_S}/test_result.txt" ]; then
                    cp "${STAGE_S}/test_result.txt" "${PT_DIR}/${RESULT_FILE_S}"
                    echo "    -> ${PT_DIR}/${RESULT_FILE_S}"
                else
                    echo "    FAILED (no small test_result)"
                fi
            else
                echo "  SKIP small matrix: $SMALL_DIR not found"
            fi
            echo ""
        done
    done
done

echo "ALL BEAM-SWEEP GREEDY EVALS DONE"
