#!/bin/bash
# =============================================================================
# Evaluation for DENSE multi-position-supervision KuaiRec checkpoints.
#   save_pt_dense_<config>_<variant>         (dense, type embeddings ON)
#   save_pt_notype_dense_<config>_<variant>  (dense, -no_type)
#   save_pt_typeauthor_fixrt_lamb0005_<variant> (dense, type + author, ONLY lamb=0.005)
#   save_pt_typemusic_fixrt_lamb0005_<variant>  (dense, type + music,  ONLY lamb=0.005)
#   save_pt_typedur_fixrt_lamb0005_<variant>    (dense, type + duration bucket, ONLY lamb=0.005)
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
    "lamb0005_consec005|0.005|0.05"
    "lamb0005_consec01|0.005|0.1"
    "lamb001|0.01|0"
    "lamb005|0.05|0"
    "lamb01|0.1|0"
)

# FAMILY|DIR_PREFIX|TYPE_FLAG
FAMILIES=(
    "type|save_pt_dense_|"
    "notype|save_pt_notype_dense_|-no_type"
)

# Side-info ablation families: type ON + one extra channel, ONLY lamb0005.
# Each needs its own side args at checkpoint load (else shapes won't match).
AUTHOR_FAMILIES=(
    "typeauthor|save_pt_typeauthor_fixrt_|"
)
AUTHOR_EXTRA="-author_file ./KuaiRec_variants/kuairec_author.txt -n_author 8369"

MUSIC_FAMILIES=(
    "typemusic|save_pt_typemusic_fixrt_|"
)
MUSIC_EXTRA="-music_file ./KuaiRec_variants/kuairec_music.txt -n_music 8494"

DUR_FAMILIES=(
    "typedur|save_pt_typedur_fixrt_|"
)
DUR_EXTRA="-dur_file ./KuaiRec_variants/kuairec_dur.txt -n_dur 8"

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_denseeval_staging"
mkdir -p "$STAGE_BASE" ./rt_dummy_for_duorec

run_eval () {
    # $1=PT_DIR $2=LATEST $3=VAR_DIR $4=EF(test) $5=EN(neg) $6=TYPE_FLAG
    # $7=MODE(topk|greedy) $8=RT_DIR("" for topk) $9=OUT $10=TAG $11=EXTRA_FLAGS
    local PT_DIR="$1" LATEST="$2" VAR_DIR="$3" EF="$4" EN="$5" TYPE_FLAG="$6"
    local MODE="$7" RT_DIR="$8" OUT="$9" TAG="${10}" EXTRA_FLAGS="${11:-}"

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
        ${TYPE_FLAG} ${EXTRA_FLAGS} ${DIV_FLAG} -t_mode ${MODE} \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i "$IN_DIR" -o "$STAGE" 2>&1 | tail -2

    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "$OUT"
        echo "    -> $OUT"
    else
        echo "    FAILED (no test_result.txt)"
    fi
}

# Two groups: dense sweeps all 9 configs; author families run only lamb0005.
for GROUP in dense author music dur; do
  case "$GROUP" in
    dense)
      CFGS=("${CONFIGS[@]}")
      FAMS=("${FAMILIES[@]}")
      EXTRA=""
      ;;
    author)
      CFGS=("lamb0005|0.005|0")
      FAMS=("${AUTHOR_FAMILIES[@]}")
      EXTRA="$AUTHOR_EXTRA"
      ;;
    music)
      CFGS=("lamb0005|0.005|0")
      FAMS=("${MUSIC_FAMILIES[@]}")
      EXTRA="$MUSIC_EXTRA"
      ;;
    dur)
      CFGS=("lamb0005|0.005|0")
      FAMS=("${DUR_FAMILIES[@]}")
      EXTRA="$DUR_EXTRA"
      ;;
  esac
for FAM in "${FAMS[@]}"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX TYPE_FLAG <<< "$FAM"
    for CFG in "${CFGS[@]}"; do
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
                "${FAM_NAME}_${SUFFIX}_${VAR}_topk_big" "$EXTRA"

            # 2. top-k, small matrix
            if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
                run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
                    "${SMALL_DIR}/test-v0.txt" \
                    "${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                    "$TYPE_FLAG" "topk" "" \
                    "${PT_DIR}/test_result_topk_small.txt" \
                    "${FAM_NAME}_${SUFFIX}_${VAR}_topk_small" "$EXTRA"
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
                    "${FAM_NAME}_${SUFFIX}_${VAR}_greedy_big" "$EXTRA"
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
                    "${FAM_NAME}_${SUFFIX}_${VAR}_greedy_small" "$EXTRA"
            fi
            echo ""
        done
    done
done
done

echo "ALL DENSE KUAIREC EVALS DONE (dense + author/music/dur@lamb0005)"
