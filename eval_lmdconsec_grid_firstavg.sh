#!/bin/bash
# =============================================================================
# lambda_c GRID EVAL — kuairec_first_average, fixed lambda=0.005, dense.
# Runs AFTER train_lmdconsec_grid_firstavg.sh.
#
# Each grid point uses its OWN separately-trained checkpoint (produced by
# train_lmdconsec_grid_firstavg.sh WITH the Sep-14 penalty indexing fix) and is
# re-decoded step-wise greedy with the SAME -lmd_consec it was trained with
# (the penalty is in the generation score q = (1-lambda)P_rel + lambda P_cov
# - lambda_c cos):
#
#   lambda_c  checkpoint suffix (type / notype)
#   0         lamb0005
#   0.001     lamb0005_consec0001
#   0.005     lamb0005_consec0005
#   0.01      lamb0005_consec001
#   0.05      lamb0005_consec005
#   0.1       lamb0005_consec01
#
# Both families: TYPE (PACER Full, save_pt_dense_*) and NOTYPE
# (save_pt_notype_dense_*). tau_o = 0.1 default everywhere (as trained).
#
# Results go to NEW filenames test_result_gridlc{,_small}.txt inside each
# checkpoint dir, so the older eval_dense_kuairec.sh files (which decoded the
# consec checkpoints with the test-time penalty OFF) are never overwritten.
# topk is not evaluated: it bypasses the scorer and cannot see lambda_c.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_lmdconsec_grid_firstavg.sh
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${CUDA_VISIBLE_DEVICES:-0}
VAR=${LMC_VAR:-kuairec_first_average}
LAMB=0.005

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

# TAG|lambda_c|DIR_SUFFIX
CONFIGS=(
    "c0|0|lamb0005"
    "c0001|0.001|lamb0005_consec0001"
    "c0005|0.005|lamb0005_consec0005"
    "c001|0.01|lamb0005_consec001"
    "c005|0.05|lamb0005_consec005"
    "c01|0.1|lamb0005_consec01"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

run_eval () {
    # $1=PT_DIR $2=LATEST $3=EF $4=EN $5=LMC $6=TYPE_FLAG $7=OUT $8=TAG
    local PT_DIR="$1" LATEST="$2" EF="$3" EN="$4" LMC="$5" TYPE_FLAG="$6" OUT="$7" TAG="$8"
    local NEWER
    NEWER=$(find "${PT_DIR}/model" -name 'duorec-*.pth' -newer "$OUT" 2>/dev/null | head -1)
    if [ -s "$OUT" ] && [ -z "$NEWER" ]; then
        echo "--- ${TAG} SKIP (up-to-date: $(basename "$OUT"))"
        return 0
    fi

    local STAGE="./save_denseeval_staging/${TAG}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"

    echo "--- ${TAG}: $(basename "$PT_DIR") epoch ${LATEST}, greedy -lamb ${LAMB} -lmd_consec ${LMC}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf "${VAR_DIR}/train-v0.txt" \
        -vf "${VAR_DIR}/valid-v0.txt" \
        -ef "$EF" -vn "$EN" -en "$EN" \
        -cat "${VAR_DIR}/kuairec_cate.txt" \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${TYPE_FLAG} -div -lamb ${LAMB} -gamma_consec 0 -lmd_consec ${LMC} \
        -t_mode greedy -start_epoch ${LATEST} -epoch_step 1 \
        -i "$RT_DIR" -o "$STAGE" 2>&1 | tail -2
    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "$OUT" && echo "    -> $OUT"
    else
        echo "    FAILED (no test_result.txt)"
    fi
}

echo "############################################################"
echo "# lambda_c GRID eval, ${VAR}, fixed lambda=${LAMB}"
echo "############################################################"

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME PREFIX TYPE_FLAG <<< "$FAM"
    for CFG in "${CONFIGS[@]}"; do
        IFS='|' read -r TAG LMC SUFFIX <<< "$CFG"
        PT_DIR="./${PREFIX}${SUFFIX}_${VAR}"

        if [ ! -d "${PT_DIR}/model" ]; then
            echo "SKIP [${FAM_NAME}/${TAG}]: missing ${PT_DIR}"
            continue
        fi
        LATEST=$(get_latest_epoch "${PT_DIR}/model")
        [ -z "$LATEST" ] && { echo "SKIP [${FAM_NAME}/${TAG}]: no checkpoint in ${PT_DIR}"; continue; }

        run_eval "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
                 "$LMC" "$TYPE_FLAG" "${PT_DIR}/test_result_gridlc.txt" \
                 "lmc_${FAM_NAME}_${TAG}_big"
        if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
            run_eval "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                     "$LMC" "$TYPE_FLAG" "${PT_DIR}/test_result_small_gridlc.txt" \
                     "lmc_${FAM_NAME}_${TAG}_small"
        fi
        echo ""
    done
done

echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
LMC_VAR="$VAR" python3 - <<'PY'
import ast, os

var = os.environ.get("LMC_VAR", "kuairec_first_average")
families = [("type", "save_pt_dense_"), ("notype", "save_pt_notype_dense_")]
grid = [("0", "lamb0005"),
        ("0.001", "lamb0005_consec0001"),
        ("0.005", "lamb0005_consec0005"),
        ("0.01", "lamb0005_consec001"),
        ("0.05", "lamb0005_consec005"),
        ("0.1", "lamb0005_consec01")]
keys = ["recall@5_f", "recall@10_f", "recall@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ild@20_f", "cc@20_f", "cs@20_f"]
header = f"{'lambda_c':<9} " + " ".join(f"{k.replace('_f',''):>11}" for k in keys)

for matrix, fn in [("SMALL matrix", "test_result_small_gridlc.txt"),
                   ("BIG matrix", "test_result_gridlc.txt")]:
    for fam, prefix in families:
        print(f"--- {fam} family, {matrix}")
        print(header)
        for lc, suffix in grid:
            path = os.path.join(f"{prefix}{suffix}_{var}", fn)
            try:
                with open(path) as f:
                    m = ast.literal_eval(f.readline().strip())
            except (FileNotFoundError, ValueError):
                print(f"{lc:<9} MISSING ({suffix})")
                continue
            print(f"{lc:<9} " + " ".join(f"{m.get(k, float('nan')):>11.4f}" for k in keys))
        print()
PY

echo "lambda_c GRID EVAL DONE"
