#!/bin/bash
# =============================================================================
# Standalone evaluation for the 2^3 EMBEDDING ABLATION (TYPE x AUTHOR x MUSIC)
# at the single lambda=0.01 operating point, KuaiRec dense checkpoints.
#
# The 8 cells of the ablation (two baselines come from the dense sweep):
#   notype      (no side info)            save_pt_notype_dense_lamb01_<variant>
#   type        (type only)               save_pt_dense_lamb01_<variant>
#   author      (notype + author)         save_pt_author_fixrt_lamb01_<variant>
#   music       (notype + music)          save_pt_music_fixrt_lamb01_<variant>
#   authormusic (notype + author + music) save_pt_authormusic_fixrt_lamb01_<variant>
#   typeauthor  (type + author)           save_pt_typeauthor_fixrt_lamb01_<variant>
#   typemusic   (type + music)            save_pt_typemusic_fixrt_lamb01_<variant>
#   typeall     (type + author + music)   save_pt_typeall_fixrt_lamb01_<variant>
#
# Train the 6 non-baseline families with:
#   nohup bash train_pt_sideinfo01_fixrt.sh <GPU_ID> > pt_sideinfo01_1000.log 2>&1 &
# (Duration is excluded: it performed poorly in the earlier lambda=0.005 study.)
#
# Same four protocols per checkpoint as eval_dense_kuairec.sh, and the same
# result filenames, so up-to-date results are SKIPped rather than recomputed:
#   top-k big matrix    -> test_result_topk.txt
#   top-k small matrix  -> test_result_topk_small.txt
#   greedy big matrix   -> test_result.txt         (needs save_rt_fix_<variant>)
#   greedy small matrix -> test_result_small.txt   (needs KuaiRec_small_eval/<variant>)
#
# Greedy decoding uses -div -lamb 0.01 -gamma_consec 0 (no -lmd_consec),
# matching the main lambda-sweep operating point.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_embedding_ablation_kuairec.sh
#   nohup bash eval_embedding_ablation_kuairec.sh > eval_ablation.log 2>&1 &
#
# Evaluate additional variants once their checkpoints exist:
#   ABL_VARIANTS="kuairec_highest_individual kuairec_first_average" \
#       bash eval_embedding_ablation_kuairec.sh
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${CUDA_VISIBLE_DEVICES:-0}
LAMB=0.01
CONSEC=0

VARIANTS=( ${ABL_VARIANTS:-kuairec_first_average} )

AUTHOR_EXTRA="-author_file ./KuaiRec_variants/kuairec_author.txt -n_author 8369"
MUSIC_EXTRA="-music_file ./KuaiRec_variants/kuairec_music.txt -n_music 8494"

# NAME|PT_DIR_PREFIX|TYPE_FLAG|EXTRA_FLAGS  (empty TYPE_FLAG = type ON)
FAMILIES=(
    "notype|save_pt_notype_dense_lamb01|-no_type|"
    "type|save_pt_dense_lamb01||"
    "author|save_pt_author_fixrt_lamb01|-no_type|${AUTHOR_EXTRA}"
    "music|save_pt_music_fixrt_lamb01|-no_type|${MUSIC_EXTRA}"
    "authormusic|save_pt_authormusic_fixrt_lamb01|-no_type|${AUTHOR_EXTRA} ${MUSIC_EXTRA}"
    "typeauthor|save_pt_typeauthor_fixrt_lamb01||${AUTHOR_EXTRA}"
    "typemusic|save_pt_typemusic_fixrt_lamb01||${MUSIC_EXTRA}"
    "typeall|save_pt_typeall_fixrt_lamb01||${AUTHOR_EXTRA} ${MUSIC_EXTRA}"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_ablation_staging"
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
        DIV_FLAG="-div -lamb ${LAMB} -gamma_consec ${CONSEC}"
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

echo "############################################################"
echo "# EMBEDDING ABLATION eval (2^3 TYPE x AUTHOR x MUSIC @ lamb=0.01)"
echo "# Variants: ${VARIANTS[*]}; families: ${#FAMILIES[@]}"
echo "############################################################"

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX TYPE_FLAG EXTRA <<< "$FAM"
    for VAR in "${VARIANTS[@]}"; do
        PT_DIR="./${DIR_PREFIX}_${VAR}"
        if [ ! -d "$PT_DIR/model" ]; then
            echo "SKIP [${FAM_NAME}/${VAR}]: missing $PT_DIR (train it first)"
            continue
        fi
        LATEST=$(get_latest_epoch "${PT_DIR}/model")
        if [ -z "$LATEST" ]; then
            echo "SKIP [${FAM_NAME}/${VAR}]: no checkpoint in $PT_DIR"
            continue
        fi

        VAR_DIR="./KuaiRec_variants/${VAR}"
        SMALL_DIR="./KuaiRec_small_eval/${VAR}"
        RT_DIR="./save_rt_fix_${VAR}"

        # 1. top-k, big matrix (pure accuracy, no RT)
        run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
            "${VAR_DIR}/test-v0.txt" \
            "${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
            "$TYPE_FLAG" "topk" "" \
            "${PT_DIR}/test_result_topk.txt" \
            "abl_${FAM_NAME}_${VAR}_topk_big" "$EXTRA"

        # 2. top-k, small matrix
        if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
            run_eval "$PT_DIR" "$LATEST" "$VAR_DIR" \
                "${SMALL_DIR}/test-v0.txt" \
                "${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt" \
                "$TYPE_FLAG" "topk" "" \
                "${PT_DIR}/test_result_topk_small.txt" \
                "abl_${FAM_NAME}_${VAR}_topk_small" "$EXTRA"
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
                "abl_${FAM_NAME}_${VAR}_greedy_big" "$EXTRA"
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
                "abl_${FAM_NAME}_${VAR}_greedy_small" "$EXTRA"
        fi
        echo ""
    done
done

echo "EMBEDDING ABLATION EVAL DONE (8 families x ${#VARIANTS[@]} variant(s), lamb=0.01)"
