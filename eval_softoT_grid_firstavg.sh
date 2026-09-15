#!/bin/bash
# =============================================================================
# T (SOFT-SELECTION TEMPERATURE) GRID EVAL — kuairec_first_average, dense,
# fixed lambda=0.01, fixed gamma_o=0.01, penalty OFF (lambda_c=0).
# Runs AFTER train_softoT_grid_firstavg.sh (and the gamma minimal rerun that
# produces the T=1.0 checkpoints, see below).
#
# Every T point is a separately TRAINED checkpoint (fixed-loss power
# selection pi = Q^(1/T)/Z); T itself is a TRAINING hyperparameter, so no T
# flag is needed at eval. All rows get the SAME greedy decode
# (-div -lamb 0.01 -gamma_consec 0 -t_mode greedy), matching the gamma table.
#
#   T     checkpoint dir                                  output tag
#   2.0   save_pt_{notype_}dense_lamb001_softo001_T2_*    softoT_T2
#   1.0   save_pt_{notype_}dense_lamb001_softo001_*       softoT_T1  (reuse*)
#   0.5   save_pt_{notype_}dense_lamb001_softo001_T05_*   softoT_T05
#   0.25  save_pt_{notype_}dense_lamb001_softo001_T025_*  softoT_T025
#   0.1   save_pt_{notype_}dense_lamb001_softo001_T01_*   softoT_T01
#
#   *T=1.0 is identical to the gamma grid's softo001 row (T defaults to 1.0).
#    Its gridorder result files were produced with exactly these decode
#    flags by eval_consecgamma_grid_firstavg.sh, so they are COPIED instead
#    of re-decoded. Run that eval first (CG_CONFIGS override including
#    o001|0.01|lamb001_softo001).
#
# Results: test_result{,_small}_softoT_<tag>.txt inside each checkpoint dir.
# topk is skipped (bypasses the scorer). Needs save_rt_fix_<variant>.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_softoT_grid_firstavg.sh
#   ST_SMALL_ONLY=1 CUDA_VISIBLE_DEVICES=0 bash eval_softoT_grid_firstavg.sh
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${CUDA_VISIBLE_DEVICES:-0}
SMALL_ONLY=${ST_SMALL_ONLY:-0}
VAR=${ST_VAR:-kuairec_first_average}
LAMB=0.01

VAR_DIR="./KuaiRec_variants/${VAR}"
SMALL_DIR="./KuaiRec_small_eval/${VAR}"
RT_DIR="./save_rt_fix_${VAR}"
NEG_BIG="${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
NEG_SMALL="${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"

FAMILIES=(
    "type|save_pt_dense_||"
    "notype|save_pt_notype_dense_|-no_type"
)

# TAG|T|DIR_SUFFIX|REUSE_SUFFIX_FOR_T1 (only the T1 row reuses gridorder)
CONFIGS=(
    "T2|2.0|lamb001_softo001_T2|"
    "T1|1.0|lamb001_softo001|gridorder"
    "T05|0.5|lamb001_softo001_T05|"
    "T025|0.25|lamb001_softo001_T025|"
    "T01|0.1|lamb001_softo001_T01|"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

reuse_or_run () {
    # $1=PT_DIR $2=LATEST $3=EF $4=EN $5=TYPE_FLAG $6=REUSE_NAME $7=OUT $8=TAG
    local PT_DIR="$1" LATEST="$2" EF="$3" EN="$4" TYPE_FLAG="$5" REUSE="$6" OUT="$7" TAG="$8"
    # T=1.0: reuse the identical-flag gridorder decode if present. The input
    # name follows the output's big/small prefix:
    #   test_result_softoT_T1.txt        <- test_result_gridorder.txt
    #   test_result_small_softoT_T1.txt  <- test_result_small_gridorder.txt
    local REUSE_IN="${OUT%softoT_${TAG}.txt}${REUSE}.txt"
    if [ -n "$REUSE" ] && [ -s "$REUSE_IN" ]; then
        cp "$REUSE_IN" "$OUT"
        echo "    -> reuse $(basename "$OUT") from $(basename "$REUSE_IN")"
        return 0
    fi
    local NEWER
    NEWER=$(find "${PT_DIR}/model" -name 'duorec-*.pth' -newer "$OUT" 2>/dev/null | head -1)
    if [ -s "$OUT" ] && [ -z "$NEWER" ]; then
        echo "--- ${TAG} SKIP (up-to-date: $(basename "$OUT"))"
        return 0
    fi

    local STAGE="./save_denseeval_staging/${TAG}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"

    echo "--- ${TAG}: $(basename "$PT_DIR") epoch ${LATEST}, greedy -lamb ${LAMB} (penalty off)"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf "${VAR_DIR}/train-v0.txt" \
        -vf "${VAR_DIR}/valid-v0.txt" \
        -ef "$EF" -vn "$EN" -en "$EN" \
        -cat "${VAR_DIR}/kuairec_cate.txt" \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${TYPE_FLAG} -div -lamb ${LAMB} -gamma_consec 0 -t_mode greedy \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i "$RT_DIR" -o "$STAGE" 2>&1 | tail -2
    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "$OUT" && echo "    -> $OUT"
    else
        echo "    FAILED (no test_result.txt)"
    fi
}

echo "############################################################"
echo "# SOFT-SELECTION T grid eval, ${VAR}, lambda=${LAMB}, gamma_o=0.01, lambda_c=0"
echo "############################################################"

if [ ! -d "${RT_DIR}/model" ]; then
    echo "ERROR: RT checkpoint missing in ${RT_DIR}/model - greedy decoding needs it. Abort."
    exit 1
fi

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME PREFIX TYPE_FLAG <<< "$FAM"
    for CFG in "${CONFIGS[@]}"; do
        IFS='|' read -r TAG TEMP SUFFIX REUSE <<< "$CFG"
        PT_DIR="./${PREFIX}${SUFFIX}_${VAR}"

        if [ ! -d "${PT_DIR}/model" ]; then
            echo "SKIP [${FAM_NAME}/${TAG}]: missing ${PT_DIR}"
            continue
        fi
        LATEST=$(get_latest_epoch "${PT_DIR}/model")
        [ -z "$LATEST" ] && { echo "SKIP [${FAM_NAME}/${TAG}]: no checkpoint in ${PT_DIR}"; continue; }

        OUT_BIG="${PT_DIR}/test_result_softoT_${TAG}.txt"
        OUT_SMALL="${PT_DIR}/test_result_small_softoT_${TAG}.txt"

        if [ "$SMALL_ONLY" = "0" ]; then
            reuse_or_run "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
                         "$TYPE_FLAG" "$REUSE" "$OUT_BIG" "softoT_${FAM_NAME}_${TAG}_big"
        fi
        if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
            reuse_or_run "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                         "$TYPE_FLAG" "$REUSE" "$OUT_SMALL" "softoT_${FAM_NAME}_${TAG}_small"
        fi
        echo ""
    done
done

echo "############################################################"
echo "# SUMMARY (small + big matrix)"
echo "############################################################"
ST_VAR="$VAR" python3 - <<'PY'
import ast, os

var = os.environ.get("ST_VAR", "kuairec_first_average")
families = [("type", "save_pt_dense_"), ("notype", "save_pt_notype_dense_")]
grid = [("2.0", "lamb001_softo001_T2", "T2"),
        ("1.0", "lamb001_softo001",    "T1"),
        ("0.5", "lamb001_softo001_T05", "T05"),
        ("0.25","lamb001_softo001_T025","T025"),
        ("0.1", "lamb001_softo001_T01", "T01")]
keys = ["recall@5_f", "recall@10_f", "recall@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ild@20_f", "cc@20_f", "cs@20_f", "MaxRun@20"]
header = f"{'T':<6} " + " ".join(f"{k.replace('_f',''):>11}" for k in keys)

for matrix, pre in [("SMALL matrix", "test_result_small_softoT_"),
                    ("BIG matrix",   "test_result_softoT_")]:
    for fam, prefix in families:
        print(f"--- {fam} family, {matrix}")
        print(header)
        for t, suffix, tag in grid:
            path = os.path.join(f"{prefix}{suffix}_{var}", f"{pre}{tag}.txt")
            try:
                with open(path) as f:
                    m = ast.literal_eval(f.readline().strip())
            except (FileNotFoundError, ValueError):
                print(f"{t:<6} MISSING ({suffix})")
                continue
            print(f"{t:<6} " + " ".join(f"{m.get(k, float('nan')):>11.4f}" for k in keys))
        print()
PY

echo "SOFT-SELECTION T GRID EVAL DONE"
