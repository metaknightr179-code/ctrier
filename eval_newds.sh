#!/bin/bash
# =============================================================================
# Evaluation for new datasets (ML1M, KuaiRand1K, MicroLens).
# Runs both topk (pure accuracy) and greedy (TRIER diversity mechanism) protocols.
#
# Usage:  bash eval_newds.sh <GPU_ID> <DATASET>
#         DATASET in {ML1M, KuaiRand1K, MicroLens}
#
# Outputs (per checkpoint dir):
#   test_result_topk.txt       (topk = full-catalog ranking, no RT)
#   test_result.txt            (greedy = RT beam + diversity blending)
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${1:?Usage: eval_newds.sh <GPU_ID> <DATASET>}
DS=${2:?DATASET must be ML1M, KuaiRand1K or MicroLens}
export CUDA_VISIBLE_DEVICES=${GPU}

case "$DS" in
  ML1M)
    N=3126; NCAT=18; DIR=./ML1M; CATE=ml1m_cate.txt; VEC=ml1m_vec.npy
    NEG="ML1M-random-sample_size=99-seed=4444.txt" ;;
  KuaiRand1K)
    N=133868; NCAT=49; DIR=./KuaiRand1K; CATE=kuairand_cate.txt; VEC=kuairand_vec.npy
    NEG="KuaiRand-random-sample_size=99-seed=4444.txt" ;;
  MicroLens)
    N=26923; NCAT=57; DIR=./MicroLens; CATE=microlens_cate.txt; VEC=microlens_vec.npy
    NEG="MicroLens-random-sample_size=99-seed=4444.txt" ;;
  *) echo "Unknown DATASET: $DS"; exit 1 ;;
esac

# Batch size: lower for large catalogs
BATCH=256
[ "$DS" = "KuaiRand1K" ] && BATCH=32

# Checkpoint prefixes (dense + non-dense)
PREFIXES=(
  "save_pt_dense_|"
  "save_pt_notype_dense_|-no_type"
  "save_pt_|"
  "save_pt_notype_|-no_type"
)

# Configs (7)
CONFIGS=( nodiv lamb0002 lamb0005 lamb0005_consec0001 lamb001 lamb005 lamb01 )

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_eval_staging_${DS}"
mkdir -p "$STAGE_BASE"
mkdir -p ./rt_dummy_for_duorec

run_eval() {
    # $1=PT_DIR  $2=LATEST  $3=TYPE_FLAG  $4=OUT  $5=TAG  $6=MODE (topk|greedy)
    local PT_DIR="$1" LATEST="$2" TYPE_FLAG="$3" OUT="$4" TAG="$5" MODE="$6"
    local STAGE="${STAGE_BASE}/${TAG}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE/model"

    echo "--- ${MODE^^} [${TAG}] epoch ${LATEST}"

    local DIV_FLAG=""
    [ "$MODE" = "greedy" ] && [ "$SUFFIX" != "nodiv" ] && DIV_FLAG="-div"

    python3 main_pt.py \
        -tf "${DIR}/train-v0.txt" \
        -vf "${DIR}/valid-v0.txt" \
        -ef "${DIR}/test-v0.txt" \
        -vn "${DIR}/${NEG}" -en "${DIR}/${NEG}" \
        -cat "${DIR}/${CATE}" -vec "${DIR}/${VEC}" \
        -n ${N} -n_cat ${NCAT} -m test -e ${LATEST} -b ${BATCH} \
        ${TYPE_FLAG} ${DIV_FLAG} -t_mode ${MODE} \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i ./rt_dummy_for_duorec -o "$STAGE" 2>&1 | tail -2

    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "$OUT"
        echo "    -> $OUT"
    else
        echo "    FAILED (no test_result.txt)"
    fi
}

# RT checkpoint dir for greedy mode
RT_DIR="./save_rt_fix_${DS}"
RT_LATEST=$(get_latest_epoch "${RT_DIR}/model")
[ -z "$RT_LATEST" ] && RT_LATEST=$(get_latest_epoch "${RT_DIR}")

for PREFIX_INFO in "${PREFIXES[@]}"; do
    IFS='|' read -r DIR_PREFIX TYPE_FLAG <<< "$PREFIX_INFO"
    for SUFFIX in "${CONFIGS[@]}"; do
        PT_DIR="./${DIR_PREFIX}${SUFFIX}_${DS}"
        [ ! -d "$PT_DIR/model" ] && { echo "SKIP: missing $PT_DIR"; continue; }
        LATEST=$(get_latest_epoch "${PT_DIR}/model")
        [ -z "$LATEST" ] && { echo "SKIP: no checkpoint in $PT_DIR"; continue; }

        TAG="${DIR_PREFIX}${SUFFIX}_${DS}"

        # Top-k eval (pure accuracy, no RT)
        run_eval "$PT_DIR" "$LATEST" "$TYPE_FLAG" \
            "${PT_DIR}/test_result_topk.txt" \
            "${TAG}_topk" "topk"

        # Greedy eval (RT beam + diversity) — only if RT exists
        if [ -n "$RT_LATEST" ] && [ "$RT_LATEST" -gt 0 ]; then
            # Symlink RT checkpoint into staging
            mkdir -p "${STAGE_BASE}/${TAG}_greedy"
            ln -sf "$(cd "${RT_DIR}/model" && pwd)/duorec-${RT_LATEST}.pth" \
                "${STAGE_BASE}/${TAG}_greedy/duorec-${RT_LATEST}.pth" 2>/dev/null

            run_eval "$PT_DIR" "$LATEST" "$TYPE_FLAG" \
                "${PT_DIR}/test_result.txt" \
                "${TAG}_greedy" "greedy"
        else
            echo "SKIP greedy: no RT checkpoint for ${DS}"
        fi
        echo ""
    done
done

echo "ALL EVALS [${DS}] DONE"
echo "Results in save_pt_*_${DS}/test_result_topk.txt and test_result.txt"
