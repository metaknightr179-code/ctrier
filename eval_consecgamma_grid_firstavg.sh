#!/bin/bash
# =============================================================================
# gamma_o GRID EVAL — kuairec_first_average, dense, fixed lambda = 0.01.
# Runs AFTER train_consecgamma_grid_firstavg.sh (the SOFT L_order grid).
#
# Every grid point is a separately TRAINED checkpoint that used the
# differentiable soft order loss gamma_o * L_order (-soft_order_loss,
# weight -lmd_softorder). All six are decoded with the SAME step-wise greedy
# flags: -div -lamb 0.01, score penalty OFF (-lmd_consec defaults to 0),
# gamma_consec passed as 0 at eval (training-only weight, inert at test).
# The soft-order flags are NOT needed at eval: L_order is a training loss.
#
# Unlike the first (invalid) grid eval, NOTHING is reused from the canonical
# test_result{,_small}.txt files: those legacy files for plain lamb001 were
# decoded with the old hardcoded lambda_c=0.01 score penalty, which made the
# gamma=0.01 row incomparable (CS@20 0.0029 vs ~0.108 for the other rows).
# Every row here gets a fresh identical-flag decode.
#
#   gamma   checkpoint
#   0       save_pt_{notype_}dense_lamb001_order0_<variant>
#   0.001   save_pt_{notype_}dense_lamb001_order0001_<variant>
#   0.005   save_pt_{notype_}dense_lamb001_order0005_<variant>
#   0.01    save_pt_{notype_}dense_lamb001_order001_<variant>
#   0.05   save_pt_{notype_}dense_lamb001_order005_<variant>
#   0.1     save_pt_{notype_}dense_lamb001_order01_<variant>
#
# Results: test_result_gridorder{,_small}.txt inside each checkpoint dir.
# topk is skipped (it bypasses the scorer; gamma_o is a training-only weight).
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

# TAG|gamma|DIR_SUFFIX (all six are soft-L_order checkpoints)
# Override for the FIXED-LOSS minimal rerun (power-annealed pi, see
# trier_pt.py soft_order_loss + check_order_gradient.py). New checkpoints are
# trained by:
#   CG_CONFIGS="softo001|0.01 softo005|0.05" \
#       bash train_consecgamma_grid_firstavg.sh 0
# and evaluated with the gamma=0 control mapped to the EXISTING zero-weight
# checkpoint lamb001_order0 (same flags, weight 0 = loss absent; never
# retrained — the train script hard-skips gamma=0):
#   CG_CONFIGS="o0|0|lamb001_order0 o001|0.01|lamb001_softo001 o005|0.05|lamb001_softo005" \
#       bash eval_consecgamma_grid_firstavg.sh
if [ -n "${CG_CONFIGS:-}" ]; then
    CONFIGS=( $CG_CONFIGS )
else
    CONFIGS=(
        "o0|0|lamb001_order0"
        "o0001|0.001|lamb001_order0001"
        "o0005|0.005|lamb001_order0005"
        "o001|0.01|lamb001_order001"
        "o005|0.05|lamb001_order005"
        "o01|0.1|lamb001_order01"
    )
fi
# Summary points derived from the same grid (passed into the python report).
CG_POINTS=""
for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG GAMMA SUFFIX <<< "$CFG"
    CG_POINTS+="${GAMMA},${SUFFIX} "
done
export CG_POINTS

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
echo "# gamma_o SOFT L_order grid eval, ${VAR}, lambda=${LAMB}, lambda_c=0"
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

        # Every row is freshly decoded with identical flags (no canonical reuse).
        run_eval "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
                 "$TYPE_FLAG" "${PT_DIR}/test_result_gridorder.txt" "ogrid_${FAM_NAME}_${TAG}_big"

        if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
            run_eval "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                     "$TYPE_FLAG" "${PT_DIR}/test_result_small_gridorder.txt" "ogrid_${FAM_NAME}_${TAG}_small"
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
spec = os.environ.get("CG_POINTS", "").strip()
if spec:
    grid = [tuple(p.split(",", 1)) for p in spec.split()]
else:
    grid = [("0", "lamb001_order0"),
            ("0.001", "lamb001_order0001"),
            ("0.005", "lamb001_order0005"),
            ("0.01", "lamb001_order001"),
            ("0.05", "lamb001_order005"),
            ("0.1", "lamb001_order01")]
keys = ["recall@5_f", "recall@10_f", "recall@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ild@20_f", "cc@20_f", "cs@20_f", "MaxRun@20"]
header = f"{'gamma':<9} " + " ".join(f"{k.replace('_f',''):>11}" for k in keys)

for matrix, fn in [("SMALL matrix", "test_result_small_gridorder.txt"),
                   ("BIG matrix", "test_result_gridorder.txt")]:
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

echo "gamma_o SOFT L_order GRID EVAL DONE"
