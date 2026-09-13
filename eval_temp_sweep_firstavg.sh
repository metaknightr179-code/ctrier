#!/bin/bash
# =============================================================================
# Eval for the tau_o TEMPERATURE SWEEP (run after train_temp_sweep_firstavg.sh).
# Fixed: variant = kuairec_first_average, TYPE family, lambda = 0.01,
# step-wise greedy decoding (-div -lamb 0.01 -gamma_consec 0, no -lmd_consec).
# Only tau_o differs; each checkpoint is re-decoded with its OWN tau_o.
#
# tau_o = 0.1 reuses save_pt_dense_lamb001_kuairec_first_average (trained with
# the original hardcoded x10 = 1/0.1 at -lamb 0.01; dense naming lamb001, NOT
# lamb01 which is -lamb 0.1); its test_result{,_small}.txt are the 0.1 cells
# and are SKIPped when up-to-date.
#
# top-k evals are intentionally skipped: that path bypasses the scorer, so
# tau_o cannot affect it (the top-k numbers are identical across the sweep).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_temp_sweep_firstavg.sh
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${CUDA_VISIBLE_DEVICES:-0}
VAR=kuairec_first_average
LAMB=0.01

VAR_DIR="./KuaiRec_variants/${VAR}"
SMALL_DIR="./KuaiRec_small_eval/${VAR}"
RT_DIR="./save_rt_fix_${VAR}"
NEG_BIG="${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
NEG_SMALL="${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"

# TAG|tau_o|PT_DIR
CONFIGS=(
    "tau005|0.05|save_pt_dense_lamb001_tau005_${VAR}"
    "tau01|0.1|save_pt_dense_lamb001_${VAR}"
    "tau02|0.2|save_pt_dense_lamb001_tau02_${VAR}"
    "tau05|0.5|save_pt_dense_lamb001_tau05_${VAR}"
    "tau1|1.0|save_pt_dense_lamb001_tau1_${VAR}"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

run_eval () {
    # $1=PT_DIR $2=LATEST $3=EF $4=EN $5=TAU $6=OUT $7=TAG
    local PT_DIR="$1" LATEST="$2" EF="$3" EN="$4" TAU="$5" OUT="$6" TAG="$7"
    local NEWER
    NEWER=$(find "${PT_DIR}/model" -name 'duorec-*.pth' -newer "$OUT" 2>/dev/null | head -1)
    if [ -s "$OUT" ] && [ -z "$NEWER" ]; then
        echo "--- ${TAG} SKIP (up-to-date: $(basename "$OUT"))"
        return 0
    fi

    local STAGE="./save_denseeval_staging/${TAG}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"

    echo "--- ${TAG}: $(basename "$PT_DIR") epoch ${LATEST}, greedy -div -lamb ${LAMB} -tau_o ${TAU}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf "${VAR_DIR}/train-v0.txt" \
        -vf "${VAR_DIR}/valid-v0.txt" \
        -ef "$EF" -vn "$EN" -en "$EN" \
        -cat "${VAR_DIR}/kuairec_cate.txt" \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        -div -lamb ${LAMB} -tau_o ${TAU} -gamma_consec 0 -t_mode greedy \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i "$RT_DIR" -o "$STAGE" 2>&1 | tail -2
    cp "${STAGE}/test_result.txt" "$OUT" && echo "    -> $OUT"
}

echo "############################################################"
echo "# tau_o TEMPERATURE SWEEP eval (${VAR}, lambda=${LAMB}, greedy)"
echo "############################################################"

for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG TAU PT_DIR <<< "$CFG"
    if [ ! -d "${PT_DIR}/model" ]; then
        echo "SKIP [${TAG}]: missing ${PT_DIR} (train it first)"
        continue
    fi
    LATEST=$(get_latest_epoch "${PT_DIR}/model")
    [ -z "$LATEST" ] && { echo "SKIP [${TAG}]: no checkpoint in ${PT_DIR}"; continue; }

    run_eval "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
             "$TAU" "${PT_DIR}/test_result.txt" "temp_${TAG}_big"
    if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
        run_eval "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                 "$TAU" "${PT_DIR}/test_result_small.txt" "temp_${TAG}_small"
    fi
    echo ""
done

echo "############################################################"
echo "# SUMMARY (small matrix)"
echo "############################################################"
python3 - <<'PY'
import ast

configs = [
    ("0.05", "save_pt_dense_lamb001_tau005_kuairec_first_average"),
    ("0.1",  "save_pt_dense_lamb001_kuairec_first_average"),
    ("0.2",  "save_pt_dense_lamb001_tau02_kuairec_first_average"),
    ("0.5",  "save_pt_dense_lamb001_tau05_kuairec_first_average"),
    ("1.0",  "save_pt_dense_lamb001_tau1_kuairec_first_average"),
]
keys = ["recall@5_f", "recall@10_f", "recall@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ild@20_f", "cc@20_f", "cs@20_f"]
print(f"{'tau_o':<6} " + " ".join(f"{k.replace('_f',''):>11}" for k in keys))
for tau, d in configs:
    path = f"{d}/test_result_small.txt"
    try:
        with open(path) as f:
            m = ast.literal_eval(f.readline().strip())
    except FileNotFoundError:
        print(f"{tau:<6} MISSING ({path})")
        continue
    print(f"{tau:<6} " + " ".join(f"{m.get(k, float('nan')):>11.4f}" for k in keys))
PY

echo "tau_o TEMPERATURE SWEEP EVAL DONE"
