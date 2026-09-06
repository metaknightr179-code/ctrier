#!/bin/bash
# =============================================================================
# GREEDY-mode evaluation (original TRIER inference) on the BIG-matrix protocol.
# Uses the full generate_by_score path: RT beam-search candidate generation +
# lambda diversity blending. This is the TRIER mechanism in action.
#
# For the pure-accuracy full-catalog ranking, use eval_topk_fixrt.sh.
#
# Full grid: 7 configs x 4 variants x 2 families (type / notype). Missing
# checkpoints are skipped, so this is safe to run during training and re-run
# later.
#
# IMPORTANT: nodiv passes "-lamb 0" explicitly. Argparse default lamb=0.5 would
# otherwise blend 50% diversity score at inference (calculate_score), making a
# "nodiv" run secretly lambda=0.5.
#
# Output (staging dir, symlinked model, never touches the checkpoint dir):
#   big matrix -> <PT_DIR>/test_result.txt   (greedy; overwritten per run)
# =============================================================================
cd "$(dirname "$0")"

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

# SUFFIX|LAMB|CONSEC (nodiv -> lamb 0)
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
    "type|save_pt_fixrt_|"
    "notype|save_pt_notype_fixrt_|-no_type"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_greedy_staging"
mkdir -p "$STAGE_BASE"

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX TYPE_FLAG <<< "$FAM"
    for CFG in "${CONFIGS[@]}"; do
        IFS='|' read -r SUFFIX LAMB CONSEC <<< "$CFG"
        for VAR in "${VARIANTS[@]}"; do
            PT_DIR="./${DIR_PREFIX}${SUFFIX}_${VAR}"
            RT_DIR="./save_rt_fix_${VAR}"
            VAR_DIR="./KuaiRec_variants/${VAR}"

            [ ! -d "$PT_DIR/model" ] && { echo "SKIP: missing $PT_DIR"; continue; }
            [ ! -d "$RT_DIR/model" ] && { echo "SKIP: missing RT $RT_DIR (needed for greedy)"; continue; }
            LATEST=$(get_latest_epoch "${PT_DIR}/model")
            [ -z "$LATEST" ] && { echo "SKIP: no checkpoint in $PT_DIR"; continue; }

            # nodiv MUST be -lamb 0 explicitly (argparse default is 0.5)
            if [ "$LAMB" == "0" ]; then
                DIV_FLAG="-lamb 0"
            else
                DIV_FLAG="-div -lamb ${LAMB} -lmd_consec ${CONSEC}"
            fi

            TAG="${FAM_NAME}_${SUFFIX}_${VAR}_big"
            STAGE="${STAGE_BASE}/${TAG}"
            rm -rf "$STAGE"; mkdir -p "$STAGE"
            ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE/model"

            echo "=============================================="
            echo "GREEDY big-matrix [${TAG}] epoch ${LATEST} (lamb=${LAMB})"
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
                ${TYPE_FLAG} ${DIV_FLAG} -t_mode greedy \
                -start_epoch ${LATEST} -epoch_step 1 \
                -i "$RT_DIR" -o "$STAGE" 2>&1 | tail -3

            if [ -f "${STAGE}/test_result.txt" ]; then
                cp "${STAGE}/test_result.txt" "${PT_DIR}/test_result.txt"
                echo "    -> ${PT_DIR}/test_result.txt"
            else
                echo "    FAILED (no test_result.txt)"
            fi
            echo ""
        done
    done
done

echo "ALL GREEDY EVALS DONE"
