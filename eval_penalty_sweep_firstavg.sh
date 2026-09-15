#!/bin/bash
# =============================================================================
# HARD INFERENCE PENALTY (lambda_c) SWEEP — inference-only, no retraining.
#
# Fixed: variant = kuairec_first_average, dense checkpoints trained at
# -lamb 0.01 (save_pt_{notype_}dense_lamb001_<variant>), greedy decoding with
# -div -lamb 0.01. We re-decode the SAME checkpoints while sweeping the
# inference-time hard adjacent penalty
#
#     score(j) = (1-lambda) rel + lambda div - lambda_c * cos(v_{j-1}, v_j)
#
# i.e. the -lmd_consec flag (see trier_pt.py calculate_score). NOTE: this is
# distinct from -gamma_consec, which weights the TRAINING loss gamma_o L_order
# and is inert at test time; here it is passed as 0.
#
#   lambda_c   output file (in the checkpoint dir)
#   0          test_result.txt / test_result_small.txt   (canonical, reused)
#   0.001      test_result_pen0001{,_small}.txt
#   0.005      test_result_pen0005{,_small}.txt
#   0.01       test_result_pen001{,_small}.txt   (paper default lambda_c=0.01)
#   0.05       test_result_pen005{,_small}.txt
#   0.1        test_result_pen01{,_small}.txt
#
# top-k is skipped (full-catalog ranking never invokes the scorer, so
# lambda_c has no effect there). Needs save_rt_fix_<variant> for greedy.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_penalty_sweep_firstavg.sh
#   nohup bash eval_penalty_sweep_firstavg.sh > eval_penalty_sweep.log 2>&1 &
#
# Families can be overridden (space-separated "name:dirprefix:typeflag"):
#   PEN_FAMILIES="typeall:save_pt_typeall_fixrt_lamb01:" \
#       bash eval_penalty_sweep_firstavg.sh
# =============================================================================
cd "$(dirname "$0")"
set -u

GPU=${CUDA_VISIBLE_DEVICES:-0}
VAR=${PEN_VAR:-kuairec_first_average}
LAMB=0.01

VAR_DIR="./KuaiRec_variants/${VAR}"
SMALL_DIR="./KuaiRec_small_eval/${VAR}"
RT_DIR="./save_rt_fix_${VAR}"
NEG_BIG="${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
NEG_SMALL="${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"

# FAMILY_NAME|DIR_PREFIX|TYPE_FLAG
#
# NOTE: lambda_c λ_c is the inference-time hard consecutive penalty. It is
# specifically designed to FIX the "frustrated regime" that appears when
# γ_o L_order is trained at 0.01 (accuracy peaks but repetition worsens —
# see Figures/tau_s_bars + gamma_o_bars in the paper). Therefore the sweep
# MUST run on checkpoints that WERE trained with L_order at γ_o=0.01 —
# i.e. the softo001 suffix. Running on the γ=0 (no-L_order) checkpoints
# produces a meaningless sweep: those checkpoints have no frustrated regime
# to fix, so λ_c looks like a random knob rather than the targeted fix it is.
#
#   save_pt_dense_lamb001_softo001_*          → PACER-Full (content + L_order trained, no type flag)
#   save_pt_notype_dense_lamb001_softo001_*   → PACER-LS  (no content, L_order trained, -no_type)
#
# If you need the γ=0 comparison (is λ_c useful even WITHOUT L_order?), run
# PEN_FAMILIES='type|save_pt_dense_lamb001|
# notype|save_pt_notype_dense_lamb001|-no_type' \
#   bash eval_penalty_sweep_firstavg.sh
if [ -n "${PEN_FAMILIES:-}" ]; then
    FAMILIES=()
    while IFS= read -r line; do [ -n "$line" ] && FAMILIES+=("$line"); done <<< "$PEN_FAMILIES"
else
    FAMILIES=(
        "PACER-Full|save_pt_dense_lamb001_softo001|"
        "PACER-LS|save_pt_notype_dense_lamb001_softo001|-no_type"
    )
fi

# TAG|lambda_c
PENALTIES=(
    "pen0|0"
    "pen0001|0.001"
    "pen0005|0.005"
    "pen001|0.01"
    "pen005|0.05"
    "pen01|0.1"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

run_eval () {
    # $1=PT_DIR $2=LATEST $3=EF $4=EN $5=TYPE_FLAG $6=LAMBDAC $7=OUT $8=TAG
    local PT_DIR="$1" LATEST="$2" EF="$3" EN="$4" TYPE_FLAG="$5" LAMBDAC="$6" OUT="$7" TAG="$8"
    local NEWER
    NEWER=$(find "${PT_DIR}/model" -name 'duorec-*.pth' -newer "$OUT" 2>/dev/null | head -1)
    if [ -s "$OUT" ] && [ -z "$NEWER" ]; then
        echo "--- ${TAG} SKIP (up-to-date: $(basename "$OUT"))"
        return 0
    fi

    local STAGE="./save_denseeval_staging/${TAG}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"

    echo "--- ${TAG}: $(basename "$PT_DIR") epoch ${LATEST}, greedy -lamb ${LAMB} -lmd_consec ${LAMBDAC}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf "${VAR_DIR}/train-v0.txt" \
        -vf "${VAR_DIR}/valid-v0.txt" \
        -ef "$EF" -vn "$EN" -en "$EN" \
        -cat "${VAR_DIR}/kuairec_cate.txt" \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${TYPE_FLAG} -div -lamb ${LAMB} -gamma_consec 0 \
        -lmd_consec ${LAMBDAC} -t_mode greedy \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i "$RT_DIR" -o "$STAGE" 2>&1 | tail -2
    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "$OUT" && echo "    -> $OUT"
    else
        echo "    FAILED (no test_result.txt)"
    fi
}

echo "############################################################"
echo "# lambda_c HARD INFERENCE PENALTY sweep, ${VAR}, lambda=${LAMB}"
echo "# Families: ${#FAMILIES[@]}; penalty points: ${#PENALTIES[@]}"
echo "############################################################"

if [ ! -d "${RT_DIR}/model" ]; then
    echo "ERROR: RT checkpoint missing in ${RT_DIR}/model - greedy decoding needs it. Abort."
    exit 1
fi

for FAM in "${FAMILIES[@]}"; do
    IFS='|' read -r FAM_NAME PREFIX TYPE_FLAG <<< "$FAM"
    PT_DIR="./${PREFIX}_${VAR}"
    if [ ! -d "${PT_DIR}/model" ]; then
        echo "SKIP [${FAM_NAME}]: missing ${PT_DIR}"
        continue
    fi
    LATEST=$(get_latest_epoch "${PT_DIR}/model")
    [ -z "$LATEST" ] && { echo "SKIP [${FAM_NAME}]: no checkpoint in ${PT_DIR}"; continue; }

    for CFG in "${PENALTIES[@]}"; do
        IFS='|' read -r TAG LAMBDAC <<< "$CFG"

        # lambda_c = 0: reuse canonical greedy files (eval_dense_kuairec.sh
        # decodes lamb001 with no -lmd_consec, i.e. penalty off).
        OUT_BIG="${PT_DIR}/test_result_${TAG}.txt"
        OUT_SMALL="${PT_DIR}/test_result_small_${TAG}.txt"
        if [ "$TAG" = "pen0" ]; then
            OUT_BIG="${PT_DIR}/test_result.txt"
            OUT_SMALL="${PT_DIR}/test_result_small.txt"
        fi

        if [ ! -s "$OUT_BIG" ]; then
            run_eval "$PT_DIR" "$LATEST" "${VAR_DIR}/test-v0.txt" "$NEG_BIG" \
                     "$TYPE_FLAG" "$LAMBDAC" "$OUT_BIG" "pen_${FAM_NAME}_${TAG}_big"
        else
            echo "--- pen_${FAM_NAME}_${TAG}_big SKIP (up-to-date: $(basename "$OUT_BIG"))"
        fi

        if [ -f "${SMALL_DIR}/test-v0.txt" ]; then
            if [ ! -s "$OUT_SMALL" ]; then
                run_eval "$PT_DIR" "$LATEST" "${SMALL_DIR}/test-v0.txt" "$NEG_SMALL" \
                         "$TYPE_FLAG" "$LAMBDAC" "$OUT_SMALL" "pen_${FAM_NAME}_${TAG}_small"
            else
                echo "--- pen_${FAM_NAME}_${TAG}_small SKIP (up-to-date: $(basename "$OUT_SMALL"))"
            fi
        fi
    done
    echo ""
done

echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
# Build families env var as "NAME:PREFIX:TYPEFLAG|NAME:PREFIX:TYPEFLAG|..."
PEN_FAM="${FAMILIES[*]}"
PEN_VAR="$VAR" PEN_FAM="$PEN_FAM" python3 - <<'PY'
import ast, os, re

var = os.environ.get("PEN_VAR", "kuairec_first_average")
fam_raw = os.environ.get("PEN_FAM", "")  # space-joined NAME|PREFIX|TYPEFLAG entries
families = []
for entry in fam_raw.split():
    parts = entry.split('|')
    if len(parts) >= 2:
        families.append((parts[0], parts[1]))

grid = [("0", "test_result.txt"),
        ("0.001", "test_result_pen0001.txt"),
        ("0.005", "test_result_pen0005.txt"),
        ("0.01", "test_result_pen001.txt"),
        ("0.05", "test_result_pen005.txt"),
        ("0.1", "test_result_pen01.txt")]
grid_small = [("0", "test_result_small.txt"),
              ("0.001", "test_result_small_pen0001.txt"),
              ("0.005", "test_result_small_pen0005.txt"),
              ("0.01", "test_result_small_pen001.txt"),
              ("0.05", "test_result_small_pen005.txt"),
              ("0.1", "test_result_small_pen01.txt")]
keys = ["recall@5_f", "recall@10_f", "recall@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ILD@20", "CC@20", "CS@20", "MaxRun@20"]
header = f"{'lambda_c':<9} " + " ".join(f"{k.replace('_f',''):>11}" for k in keys)

for matrix, g in [("SMALL matrix", grid_small), ("BIG matrix", grid)]:
    for fam_name, prefix in families:
        d = f"{prefix}_{var}"
        print(f"--- {fam_name}, {matrix} ({d})")
        print(header)
        for lc, fn in g:
            path = os.path.join(d, fn)
            try:
                with open(path) as f:
                    m = ast.literal_eval(f.readline().strip())
            except (FileNotFoundError, ValueError, SyntaxError):
                print(f"{lc:<9} MISSING ({fn})")
                continue
            row = []
            for k in keys:
                v = m.get(k)
                row.append(f"{v:>11.4f}" if isinstance(v, (int, float)) else f"{'--':>11}")
            print(f"{lc:<9} " + " ".join(row))
        print()
PY

echo "lambda_c HARD INFERENCE PENALTY SWEEP DONE"
echo "LaTeX table: python3 analyze_results.py --penalty_sweep --proto small"
echo "             -> penalty_sweep_first_average_small.tex"
