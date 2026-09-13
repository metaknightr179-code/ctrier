#!/bin/bash
# =============================================================================
# SIX-CELL COMPONENT ABLATION (Content x Order-loss x Order-score),
# one variant (kuairec_first_average) at the single lambda = 0.01 operating
# point. All checkpoints already exist from the dense lambda sweep; this
# script only reuses their result files or runs the ONE new eval combination
# (TRIER-S: nodiv checkpoint decoded with the diverse greedy scorer).
#
#   Cell        checkpoint (First-Average)            Content OrderLoss OrderScore
#   TRIER       notype_dense_nodiv                      no      no        no
#   TRIER-C     dense_nodiv (type)                      yes     no        no
#   TRIER-L     notype_dense_lamb01  (topk eval)        no      yes       no
#   TRIER-S     notype_dense_nodiv  (greedy lamb=.01)   no      no        yes   <- NEW eval
#   PACER-LS    notype_dense_lamb01 (greedy lamb=.01)   no      yes       yes
#   PACER-Full  dense_lamb01        (greedy lamb=.01)   yes     yes       yes
#
# "Order loss" entered during TRAINING (-div -lamb 0.01 vs nodiv); "Order
# score" is the inference-time diverse greedy decoder (-div -lamb 0.01 vs
# relevance-only topk). Existing, bit-identical result files are copied;
# only missing combinations are actually evaluated.
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
OUTDIR="./sixcell_firstavg"
mkdir -p "$OUTDIR" ./save_denseeval_staging ./rt_dummy_for_duorec

VAR_DIR="./KuaiRec_variants/${VAR}"
SMALL_DIR="./KuaiRec_small_eval/${VAR}"
RT_DIR="./save_rt_fix_${VAR}"
NEG_BIG="${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
NEG_SMALL="${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"

# NAME|PT_DIR|TYPE_FLAG|MODE(topk|greedy)
CELLS=(
    "TRIER|save_pt_notype_dense_nodiv_${VAR}|-no_type|topk"
    "TRIER-C|save_pt_dense_nodiv_${VAR}||topk"
    "TRIER-L|save_pt_notype_dense_lamb01_${VAR}|-no_type|topk"
    "TRIER-S|save_pt_notype_dense_nodiv_${VAR}|-no_type|greedy"
    "PACER-LS|save_pt_notype_dense_lamb01_${VAR}|-no_type|greedy"
    "PACER-Full|save_pt_dense_lamb01_${VAR}||greedy"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

# Copy a canonical existing result file when it matches the cell exactly;
# returns 1 when an actual eval run is needed.
reuse_if_present () {
    local SRC="$1" DST="$2"
    if [ -s "$SRC" ]; then
        cp "$SRC" "$DST"
        echo "    -> reuse $(basename "$DST") from ${SRC#./}"
        return 0
    fi
    return 1
}

run_eval () {
    # $1=PT_DIR $2=LATEST $3=EF $4=EN $5=TYPE_FLAG $6=MODE $7=OUT
    local PT_DIR="$1" LATEST="$2" EF="$3" EN="$4" TYPE_FLAG="$5" MODE="$6" OUT="$7"
    local STAGE="./save_denseeval_staging/sixcell_$(basename "$PT_DIR")_$(basename "$OUT")"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"

    local DIV_FLAG="-lamb 0" IN_DIR="./rt_dummy_for_duorec"
    if [ "$MODE" = "greedy" ]; then
        DIV_FLAG="-div -lamb ${LAMB} -gamma_consec 0"
        IN_DIR="$RT_DIR"
    fi

    echo "    running ${MODE} eval: $(basename "$PT_DIR") epoch ${LATEST}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf "${VAR_DIR}/train-v0.txt" \
        -vf "${VAR_DIR}/valid-v0.txt" \
        -ef "$EF" -vn "$EN" -en "$EN" \
        -cat "${VAR_DIR}/kuairec_cate.txt" \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${TYPE_FLAG} ${DIV_FLAG} -t_mode ${MODE} \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i "$IN_DIR" -o "$STAGE" 2>&1 | tail -2
    cp "${STAGE}/test_result.txt" "$OUT" && echo "    -> $(basename "$OUT")"
}

echo "############################################################"
echo "# SIX-CELL ABLATION (${VAR}, lambda=${LAMB})"
echo "############################################################"

for CELL in "${CELLS[@]}"; do
    IFS='|' read -r NAME PT_DIR TYPE_FLAG MODE <<< "$CELL"
    mkdir -p "${OUTDIR}/${NAME}"
    echo ""
    echo "== ${NAME}  [${MODE}, $( [ -n "$TYPE_FLAG" ] && echo notype || echo type )]  ${PT_DIR}"

    if [ ! -d "${PT_DIR}/model" ]; then
        echo "    MISSING checkpoint dir ${PT_DIR} — train it first"
        continue
    fi
    LATEST=$(get_latest_epoch "${PT_DIR}/model")

    # ---- big matrix ----
    OUT_BIG="${OUTDIR}/${NAME}/test_result_big.txt"
    if [ "$MODE" = "topk" ]; then
        reuse_if_present "${PT_DIR}/test_result_topk.txt" "$OUT_BIG" \
            || run_eval "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
                        "$TYPE_FLAG" topk "$OUT_BIG"
    else
        reuse_if_present "${PT_DIR}/test_result.txt" "$OUT_BIG" \
            || run_eval "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
                        "$TYPE_FLAG" greedy "$OUT_BIG"
    fi

    # ---- small matrix ----
    OUT_SMALL="${OUTDIR}/${NAME}/test_result_small.txt"
    if [ "$MODE" = "topk" ]; then
        reuse_if_present "${PT_DIR}/test_result_topk_small.txt" "$OUT_SMALL" \
            || run_eval "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                        "$TYPE_FLAG" topk "$OUT_SMALL"
    else
        reuse_if_present "${PT_DIR}/test_result_small.txt" "$OUT_SMALL" \
            || run_eval "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                        "$TYPE_FLAG" greedy "$OUT_SMALL"
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
        "ild@20_f", "cc@20_f", "cs@20_f"]
width = max(map(len, order))
print(f"{'cell':<{width}} " + " ".join(f"{k.replace('_f',''):>11}" for k in keys))
for name in order:
    path = os.path.join(outdir, name, "test_result_small.txt")
    if not os.path.exists(path):
        print(f"{name:<{width}} MISSING")
        continue
    with open(path) as f:
        d = ast.literal_eval(f.readline().strip())
    print(f"{name:<{width}} " + " ".join(f"{d.get(k, float('nan')):>11.4f}" for k in keys))
PY

echo "SIX-CELL ABLATION DONE -> ${OUTDIR}/"
