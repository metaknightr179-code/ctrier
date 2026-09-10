#!/bin/bash
# =============================================================================
# Small-matrix (canonical KuaiRec) evaluation of the fixed-RT PT checkpoints.
#
# FULL GRID: 7 lambda configs x 4 variants x 2 families (type / notype) = 56
# evals; missing checkpoint dirs are skipped automatically.
#
# Protocol: models were TRAINED on big-matrix-derived data (KuaiRec_variants/);
# here they are TESTED on the small-matrix leave-last-out splits in
# KuaiRec_small_eval/<variant>/. Item IDs are raw KuaiRec video_ids, so the
# same 0..10727 item space and kuairec_vec.npy apply.
#
# Greedy inference (RT beam generation + lambda diversity blending), matching
# the training-time flags of each checkpoint:
#   - type   checkpoints (save_pt_fixrt_*):        type embeddings ON
#   - notype checkpoints (save_pt_notype_fixrt_*): -no_type
#   - nodiv:  no -div flag;  lamb>0 configs: -div -lamb X [-lmd_consec Y]
#
# Results are written to <PT_DIR>/test_result_small.txt. The original
# test_result.txt (big-matrix results) is NOT touched: main_pt.py hardcodes its
# output filename, so the eval runs with -o pointing at a staging dir whose
# model/ is a symlink to the real checkpoint dir, and the produced
# test_result.txt is copied to test_result_small.txt afterwards.
#
# Usage: bash eval_small_fixrt.sh  (sequential, ~1-4 min per checkpoint)
# =============================================================================
cd "$(dirname "$0")"

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

# SUFFIX|LAMB|LMD_CONSEC
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
    "author|save_pt_author_fixrt_|-no_type -author_file ./KuaiRec_variants/kuairec_author.txt -n_author 8369"
    "typeauthor|save_pt_typeauthor_fixrt_|-author_file ./KuaiRec_variants/kuairec_author.txt -n_author 8369"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_smalleval_staging"
mkdir -p "$STAGE_BASE"

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX TYPE_FLAG <<< "$FAM"

    for CFG in "${CONFIGS[@]}"; do
        IFS='|' read -r SUFFIX LAMB CONSEC <<< "$CFG"

        for VAR in "${VARIANTS[@]}"; do
            PT_DIR="./${DIR_PREFIX}${SUFFIX}_${VAR}"
            RT_DIR="./save_rt_fix_${VAR}"
            SMALL_DIR="./KuaiRec_small_eval/${VAR}"

            if [ ! -d "$PT_DIR/model" ]; then echo "SKIP: missing $PT_DIR"; continue; fi
            if [ ! -d "$RT_DIR/model" ]; then echo "SKIP: missing RT $RT_DIR ($PT_DIR)"; continue; fi
            if [ ! -f "${SMALL_DIR}/test-v0.txt" ]; then echo "SKIP: missing ${SMALL_DIR}/test-v0.txt"; continue; fi

            LATEST=$(get_latest_epoch "${PT_DIR}/model")
            if [ -z "$LATEST" ]; then echo "SKIP: no checkpoint in $PT_DIR"; continue; fi

            # div flag: div_loss was active in training for all lambda>0 configs.
            # nodiv MUST pass -lamb 0 explicitly: argparse default lamb=0.5 would
            # otherwise blend 50% diversity score at inference (calculate_score),
            # making "nodiv greedy" secretly a lambda=0.5 run.
            if [ "$LAMB" == "0" ]; then
                DIV_FLAG="-lamb 0"
            else
                DIV_FLAG="-div -lamb ${LAMB} -lmd_consec ${CONSEC}"
            fi

            # staging dir: model/ symlinked to the real checkpoint dir so the eval
            # loads the right weights but writes results outside the original dir
            STAGE="${STAGE_BASE}/${FAM_NAME}_${SUFFIX}_${VAR}"
            rm -rf "$STAGE"
            mkdir -p "$STAGE"
            ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE/model"

            echo "=============================================="
            echo "SMALL-matrix eval [${FAM_NAME}]: ${SUFFIX} / ${VAR} (epoch ${LATEST}, lamb=${LAMB})"
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
                ${TYPE_FLAG} ${DIV_FLAG} -t_mode greedy \
                -start_epoch ${LATEST} -epoch_step 1 \
                -i ${RT_DIR} -o ${STAGE} 2>&1 | tee "eval_small_${FAM_NAME}_${SUFFIX}_${VAR}.log"

            if [ -f "${STAGE}/test_result.txt" ]; then
                cp "${STAGE}/test_result.txt" "${PT_DIR}/test_result_small.txt"
                echo "=== [${FAM_NAME}] ${SUFFIX}/${VAR} done -> ${PT_DIR}/test_result_small.txt ==="
            else
                echo "=== [${FAM_NAME}] ${SUFFIX}/${VAR} FAILED (no test_result.txt produced) ==="
            fi
            echo ""
        done
    done
done

echo "ALL SMALL-MATRIX FIXRT EVALS DONE"
