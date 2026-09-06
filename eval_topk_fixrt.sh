#!/bin/bash
# =============================================================================
# Top-k (full-catalog ranking) evaluation of fixed-RT PT checkpoints — the
# PURE-ACCURACY protocol.
#
# Why: greedy eval (eval_greedy_fixrt.sh / eval_small_fixrt.sh) generates
# candidates via RT beam search and blends diversity scores (-div/-lamb at
# inference), which caps accuracy by the RT candidate pool. Top-k eval ranks
# the ENTIRE 10k catalog directly from the PT encoder's last-token logits:
#   - RT model is NEVER invoked (test_forward(step_by_step=False))
#   - no diversity score blending
# This is the same inference protocol as DuoRec/SASRec/GRU4Rec, so accuracy
# is directly comparable.
#
# Accuracy-oriented TRIER = nodiv checkpoint (trained with CE + contrastive
# NCE only) + topk eval. lambda>0 checkpoints are also eval'd here to show
# the training-time accuracy cost of the diversity loss.
#
# Outputs (do NOT touch greedy results):
#   big   matrix -> <PT_DIR>/test_result_topk.txt
#   small matrix -> <PT_DIR>/test_result_topk_small.txt
#
# Usage: bash eval_topk_fixrt.sh   (sequential; ~1 min per checkpoint)
# =============================================================================
cd "$(dirname "$0")"

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

# SUFFIX (all 7; nodiv is the pure-accuracy headline)
CONFIGS=( nodiv lamb0002 lamb0005 lamb0005_consec0001 lamb001 lamb005 lamb01 )

# FAMILY|DIR_PREFIX|TYPE_FLAG
FAMILIES=(
    "type|save_pt_fixrt_|"
    "notype|save_pt_notype_fixrt_|-no_type"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_topk_staging"
mkdir -p "$STAGE_BASE"

run_eval () {
    # $1 = PT_DIR, $2 = latest epoch, $3 = test file, $4 = neg file, $5 = type flag,
    # $6 = out result path, $7 = tag
    local PT_DIR="$1" LATEST="$2" EF="$3" EN="$4" TYPE_FLAG="$5" OUT="$6" TAG="$7"
    local STAGE="${STAGE_BASE}/${TAG}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE/model"

    echo "--- TOPK [${TAG}] $(basename "$PT_DIR") epoch ${LATEST}"
    python3 main_pt.py \
        -tf ./KuaiRec_variants/kuairec_highest_individual/train-v0.txt \
        -vf ./KuaiRec_variants/kuairec_highest_individual/valid-v0.txt \
        -ef "$EF" -vn "$EN" -en "$EN" \
        -cat ./KuaiRec_variants/kuairec_highest_individual/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${TYPE_FLAG} -t_mode topk \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i ./rt_dummy_for_duorec -o "$STAGE" 2>&1 | tail -2

    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "$OUT"
        echo "    -> $OUT"
    else
        echo "    FAILED (no test_result.txt)"
    fi
}

mkdir -p ./rt_dummy_for_duorec

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX TYPE_FLAG <<< "$FAM"
    for SUFFIX in "${CONFIGS[@]}"; do
        for VAR in "${VARIANTS[@]}"; do
            PT_DIR="./${DIR_PREFIX}${SUFFIX}_${VAR}"
            [ ! -d "$PT_DIR/model" ] && { echo "SKIP: missing $PT_DIR"; continue; }
            LATEST=$(get_latest_epoch "${PT_DIR}/model")
            [ -z "$LATEST" ] && { echo "SKIP: no checkpoint in $PT_DIR"; continue; }

            VAR_DIR="./KuaiRec_variants/${VAR}"
            SMALL_DIR="./KuaiRec_small_eval/${VAR}"

            # big-matrix protocol
            run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
                "${VAR_DIR}/test-v0.txt" \
                "${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                "$TYPE_FLAG" \
                "${PT_DIR}/test_result_topk.txt" \
                "${FAM_NAME}_${SUFFIX}_${VAR}_big"

            # small-matrix protocol (canonical KuaiRec)
            if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
                run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
                    "${SMALL_DIR}/test-v0.txt" \
                    "${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                    "$TYPE_FLAG" \
                    "${PT_DIR}/test_result_topk_small.txt" \
                    "${FAM_NAME}_${SUFFIX}_${VAR}_small"
            else
                echo "SKIP small: ${SMALL_DIR}/test-v0.txt missing"
            fi
            echo ""
        done
    done
done

echo "ALL TOPK EVALS DONE"
