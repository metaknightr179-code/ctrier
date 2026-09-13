#!/bin/bash
# =============================================================================
# gamma_consec GRID EVAL — kuairec_first_average, dense, fixed lambda = 0.01.
# Runs AFTER train_consecgamma_grid_firstavg.sh.
#
# Every grid point is a separately TRAINED checkpoint (the loss weight
# gamma_consec shapes the diverse-token learning signal). All are decoded with
# the SAME step-wise greedy flags: -div -lamb 0.01, score penalty OFF
# (-lmd_consec defaults to 0). gamma_consec is a training-only weight and is
# passed as 0 at eval (inert; matches the existing dense-sweep evals).
#
#   gamma   checkpoint
#   0       save_pt_{notype_}dense_lamb001_consec0_<variant>     (new)
#   0.001   save_pt_{notype_}dense_lamb001_consec0001_<variant>  (new)
#   0.005   save_pt_{notype_}dense_lamb001_consec0005_<variant>  (new)
#   0.01    plain save_pt_{notype_}dense_lamb001_<variant>
#   0.05    save_pt_{notype_}dense_lamb001_consec005_<variant>   (new)
#   0.1     save_pt_{notype_}dense_lamb001_consec01_<variant>    (new)
#
# Results: test_result_gridgamma{,_small}.txt inside each checkpoint dir.
# The 0.01 point reuses the canonical dense-sweep files when present.
# topk is skipped (it bypasses the scorer; gamma is a training-only weight).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_consecgamma_grid_firstavg.sh
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${CUDA_VISIBLE_DEVICES:-0}
VAR=${CG_VAR:-kuairec_first_average}
LAMB=0.01

VAR_DIR="./KuaiRec_variants/${VAR}"
SMALL_DIR="./KuaiRec_small_eval/${VAR}"
RT_DIR="./save_rt_fix_${VAR}"
NEG_BIG="${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
NEG_SMALL="${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"

# FAMILY|DIR_PREFIX|TYPE_FLAG
FAMILIES=(
    "type|save_pt_dense_||"
    "notype|save_pt_notype_dense_|-no_type"
)

# TAG|gamma|DIR_SUFFIX (plain lamb001 is the gamma=0.01 default-weight model)
CONFIGS=(
    "c0|0|lamb001_consec0"
    "c0001|0.001|lamb001_consec0001"
    "c0005|0.005|lamb001_consec0005"
    "c001|0.01|lamb001"
    "c005|0.05|lamb001_consec005"
    "c01|0.1|lamb001_consec01"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

run_eval () {
    # $1=PT_DIR $2=LATEST $3=EF $4=EN $5=TYPE_FLAG $6=OUT $7=TAG
    local PT_DIR="$1" LATEST="$2" EF="$3" EN="$4" TYPE_FLAG="$5" OUT="$6" TAG="$7"
    local NEWER
    NEWER=$(find "${PT_DIR}/model" -name 'duorec-*.pth' -newer "$OUT" 2>/dev/null | head -1)
    if [ -s "$OUT" ] && [ -z "$NEWER" ]; then
        echo "--- ${TAG} SKIP (up-to-date: $(basename "$OUT"))"
        return 0
    fi

    local STAGE="./save_denseeval_staging/${TAG}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"

    echo "--- ${TAG}: $(basename "$PT_DIR") epoch ${LATEST}, greedy -lamb ${LAMB} (score penalty off)"
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
echo "# gamma_consec LOSS-WEIGHT grid eval, ${VAR}, lambda=${LAMB}"
echo "############################################################"

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME PREFIX TYPE_FLAG <<< "$FAM"
    for CFG in "${CONFIGS[@]}"; do
        IFS='|' read -r TAG GAMMA SUFFIX <<< "$CFG"
        PT_DIR="./${PREFIX}${SUFFIX}_${VAR}"

        if [ ! -d "${PT_DIR}/model" ]; then
            echo "SKIP [${FAM_NAME}/${TAG}]: missing ${PT_DIR}"
            continue
        fi
        LATEST=$(get_latest_epoch "${PT_DIR}/model")
        [ -z "$LATEST" ] && { echo "SKIP [${FAM_NAME}/${TAG}]: no checkpoint in ${PT_DIR}"; continue; }

        # gamma=0.01 = plain lamb001: reuse canonical dense-sweep greedy results
        OUT_BIG="${PT_DIR}/test_result_gridgamma.txt"
        if [ "$TAG" = "c001" ] && [ -s "${PT_DIR}/test_result.txt" ] && [ ! -s "$OUT_BIG" ]; then
            cp "${PT_DIR}/test_result.txt" "$OUT_BIG"
            echo "--- ${FAM_NAME}/c001 big: reuse test_result.txt"
        else
            run_eval "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
                     "$TYPE_FLAG" "$OUT_BIG" "cgrid_${FAM_NAME}_${TAG}_big"
        fi

        OUT_SMALL="${PT_DIR}/test_result_small_gridgamma.txt"
        if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
            if [ "$TAG" = "c001" ] && [ -s "${PT_DIR}/test_result_small.txt" ] && [ ! -s "$OUT_SMALL" ]; then
                cp "${PT_DIR}/test_result_small.txt" "$OUT_SMALL"
                echo "--- ${FAM_NAME}/c001 small: reuse test_result_small.txt"
            else
                run_eval "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                         "$TYPE_FLAG" "$OUT_SMALL" "cgrid_${FAM_NAME}_${TAG}_small"
            fi
        fi
        echo ""
    done
done

echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
CG_VAR="$VAR" python3 - <<'PY'
import ast, os

var = os.environ.get("CG_VAR", "kuairec_first_average")
families = [("type", "save_pt_dense_"), ("notype", "save_pt_notype_dense_")]
grid = [("0", "lamb001_consec0"),
        ("0.001", "lamb001_consec0001"),
        ("0.005", "lamb001_consec0005"),
        ("0.01", "lamb001"),
        ("0.05", "lamb001_consec005"),
        ("0.1", "lamb001_consec01")]
keys = ["recall@5_f", "recall@10_f", "recall@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ild@20_f", "cc@20_f", "cs@20_f"]
header = f"{'gamma':<9} " + " ".join(f"{k.replace('_f',''):>11}" for k in keys)

for matrix, fn in [("SMALL matrix", "test_result_small_gridgamma.txt"),
                   ("BIG matrix", "test_result_gridgamma.txt")]:
    for fam, prefix in families:
        print(f"--- {fam} family, {matrix}")
        print(header)
        for gc, suffix in grid:
            path = os.path.join(f"{prefix}{suffix}_{var}", fn)
            try:
                with open(path) as f:
                    m = ast.literal_eval(f.readline().strip())
            except (FileNotFoundError, ValueError):
                print(f"{gc:<9} MISSING ({suffix})")
                continue
            print(f"{gc:<9} " + " ".join(f"{m.get(k, float('nan')):>11.4f}" for k in keys))
        print()
PY

echo "gamma_consec GRID EVAL DONE"
