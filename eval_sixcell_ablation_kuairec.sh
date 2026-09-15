#!/bin/bash
# =============================================================================
# SIX-CELL COMPONENT ABLATION (Content x Order-loss x Order-score),
# kuairec_first_average at the lambda = 0.01 operating point.
#
# All six cells use the SAME step-wise greedy diverse decoder
# (-div -lamb 0.01 -t_mode greedy), so adjacent columns differ in exactly ONE
# factor:
#   * Content      : type embeddings present (dense) vs -no_type
#   * Order loss L : fixed differentiable L_order at training, gamma_o=0.01
#                    (checkpoint suffix lamb001_softo001) vs absent (lamb001)
#   * Order score S: inference adjacent penalty -lmd_consec 0.01 (=lambda_c)
#                    vs 0
#
#   Cell        checkpoint (save_pt_*)              Content OrderLoss OrderScore
#   TRIER       notype_dense_lamb001                 no      no        no
#   TRIER-C     dense_lamb001 (type)                 yes     no        no
#   TRIER-L     notype_dense_lamb001_softo001        no      yes       no
#   TRIER-S     notype_dense_lamb001                 no      no        yes
#   PACER-LS    notype_dense_lamb001_softo001        no      yes       yes
#   PACER-Full  dense_lamb001_softo001               yes     yes       yes
#
# Requirements / provenance:
#   * lamb001 dirs: canonical dense lambda=0.01 checkpoints.
#   * lamb001_softo001 dirs: FIXED power-annealed soft L_order at gamma=0.01,
#     produced by CG_FIXED=1 train_consecgamma_grid_firstavg.sh (or the
#     CG_CONFIGS="softo001|0.01" minimal subset). These replace the earlier
#     nodiv/topk cells: every cell is now decoded with the diverse scorer, so
#     the S column isolates lambda_c cleanly.
#   * save_rt_fix_<variant>: greedy decoding needs the frozen RT checkpoint.
# All cells are ALWAYS freshly decoded here (canonical greedy files are
# penalty-off and also predate MaxRun@k, so they must not be reused).
#
# Knobs:
#   SIXCELL_ORDER_SUF=lamb001_softo005  ... L column at another fixed gamma
#   SIXCELL_BASE_SUF=lamb001            ... base suffix
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_sixcell_ablation_kuairec.sh
#
# Outputs: ./sixcell_firstavg/<CELL>/test_result_{big,small}.txt
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${CUDA_VISIBLE_DEVICES:-0}
VAR=kuairec_first_average
LAMB=0.01
LMD_ON=${SIXCELL_LMD:-0.01}
BASE_SUF=${SIXCELL_BASE_SUF:-lamb001}
LOSS_SUF=${SIXCELL_ORDER_SUF:-lamb001_softo001}
OUTDIR="./sixcell_firstavg"
mkdir -p "$OUTDIR" ./save_denseeval_staging

VAR_DIR="./KuaiRec_variants/${VAR}"
SMALL_DIR="./KuaiRec_small_eval/${VAR}"
RT_DIR="./save_rt_fix_${VAR}"
NEG_BIG="${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
NEG_SMALL="${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"

# NAME|TYPE(type/notype)|L_order(1/0)|OrderScore(1/0)
CELLS=(
    "TRIER|notype|0|0"
    "TRIER-C|type|0|0"
    "TRIER-L|notype|1|0"
    "TRIER-S|notype|0|1"
    "PACER-LS|notype|1|1"
    "PACER-Full|type|1|1"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

run_eval () {
    # $1=PT_DIR $2=LATEST $3=EF $4=EN $5=TYPE_FLAG $6=LMD_CONSEC $7=OUT $8=TAG
    local PT_DIR="$1" LATEST="$2" EF="$3" EN="$4" TYPE_FLAG="$5" LMD="$6" OUT="$7" TAG="$8"
    local STAGE="./save_denseeval_staging/sixcell_${TAG}_$(basename "$OUT")"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"

    echo "    running greedy eval: $(basename "$PT_DIR") epoch ${LATEST}, lambda_c=${LMD}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf "${VAR_DIR}/train-v0.txt" \
        -vf "${VAR_DIR}/valid-v0.txt" \
        -ef "$EF" -vn "$EN" -en "$EN" \
        -cat "${VAR_DIR}/kuairec_cate.txt" \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${TYPE_FLAG} -div -lamb ${LAMB} -gamma_consec 0 -lmd_consec ${LMD} -t_mode greedy \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i "$RT_DIR" -o "$STAGE" 2>&1 | tail -2
    cp "${STAGE}/test_result.txt" "$OUT" && echo "    -> $(basename "$OUT")"
}

echo "############################################################"
echo "# SIX-CELL ABLATION (${VAR}, lambda=${LAMB})"
echo "# base=${BASE_SUF}  L_order=${LOSS_SUF}  lambda_c=${LMD_ON}"
echo "############################################################"

if [ ! -d "${RT_DIR}/model" ]; then
    echo "ERROR: RT checkpoint missing in ${RT_DIR}/model - greedy decoding needs it. Abort."
    exit 1
fi

for CELL in "${CELLS[@]}"; do
    IFS='|' read -r NAME CTYPE HAS_L HAS_S <<< "$CELL"
    mkdir -p "${OUTDIR}/${NAME}"

    if [ "$CTYPE" = "notype" ]; then
        TYPE_FLAG="-no_type"; DIR_MID="notype_"; CT_LABEL="notype"
    else
        TYPE_FLAG=""; DIR_MID=""; CT_LABEL="type"
    fi
    SUF="$BASE_SUF"; [ "$HAS_L" = "1" ] && SUF="$LOSS_SUF"
    LMD="0"; [ "$HAS_S" = "1" ] && LMD="$LMD_ON"
    PT_DIR="./save_pt_${DIR_MID}dense_${SUF}_${VAR}"

    echo ""
    echo "== ${NAME}  [${CT_LABEL}, L_order=${HAS_L}, lambda_c=${LMD}]  ${PT_DIR}"

    if [ ! -d "${PT_DIR}/model" ]; then
        echo "    MISSING checkpoint dir ${PT_DIR} — train it first"
        continue
    fi
    LATEST=$(get_latest_epoch "${PT_DIR}/model")
    [ -z "$LATEST" ] && { echo "    MISSING checkpoint .pth in ${PT_DIR}"; continue; }

    # ---- big matrix ----
    run_eval "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
                "$TYPE_FLAG" "$LMD" "${OUTDIR}/${NAME}/test_result_big.txt" "${NAME}_big"

    # ---- small matrix ----
    if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
        run_eval "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                    "$TYPE_FLAG" "$LMD" "${OUTDIR}/${NAME}/test_result_small.txt" "${NAME}_small"
    fi
done

echo ""
echo "############################################################"
echo "# SUMMARY (small matrix, first line = aggregate metrics)"
echo "############################################################"
python3 - <<'PY'
import ast, os

outdir = "sixcell_firstavg"
order = ["TRIER", "TRIER-C", "TRIER-L", "TRIER-S", "PACER-LS", "PACER-Full"]
keys = ["recall@5_f", "recall@10_f", "recall@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ILD@20", "CC@20", "CS@20", "MaxRun@20"]
width = max(map(len, order))
print(f"{'cell':<{width}} " + " ".join(f"{k.replace('_f',''):>11}" for k in keys))
for name in order:
    path = os.path.join(outdir, name, "test_result_small.txt")
    if not os.path.exists(path):
        print(f"{name:<{width}} MISSING")
        continue
    with open(path) as f:
        d = ast.literal_eval(f.readline().strip())
    row = [f"{d[k]:>11.4f}" if isinstance(d.get(k), (int, float)) else f"{'--':>11}"
           for k in keys]
    print(f"{name:<{width}} " + " ".join(row))
PY

echo "SIX-CELL ABLATION DONE -> ${OUTDIR}/"
