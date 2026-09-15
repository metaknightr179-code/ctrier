#!/bin/bash
# =============================================================================
# 3-SEED SIX-CELL EVALUATION — kuairec_first_average, greedy diverse decode.
#
# For each of the six cells, evaluates EVERY PT dir matching the suffix across
# seeds (default seeds 1 2 3; plus any legacy non-suffixed dir found on disk),
# then aggregates mean±std per metric. Output LaTeX snippet to console +
# sixcell_firstavg/stats/mean_std_{small,big}.txt for the paper table.
#
# Requirements before running:
#   * train_sixcell_3seed.sh 0          trained all 4 configs × 3 seeds
#   * save_rt_fix_kuairec_first_average exists (shared RT)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval_sixcell_3seed.sh
#   SEEDS="1 2 3" CUDA_VISIBLE_DEVICES=0 bash eval_sixcell_3seed.sh
# =============================================================================
set -u
GPU=${CUDA_VISIBLE_DEVICES:-0}
VAR=kuairec_first_average
LAMB=0.01; LMD_ON=${SIXCELL_LMD:-0.01}
DIR=./KuaiRec_variants/${VAR}
RT_OUT=save_rt_fix_${VAR}
SEEDS=${SEEDS:-"1 2"}
OUTDIR=./sixcell_firstavg_3seed
mkdir -p ${OUTDIR}/staging ${OUTDIR}/stats

NEG_BIG="${DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
NEG_SMALL="./KuaiRec_small_eval/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt"
HAS_SMALL=""
[ -f "$NEG_SMALL" ] && HAS_SMALL=1

# Six cells -> lookup PT suffix (base or softo), with which seed family
# NAME|TYPE_FLAG|BASE_SUF|LOSS_SUF|HAS_S
CELLS=(
  "TRIER|-no_type|notype_dense_lamb001_order0|notype_dense_lamb001_softo001|0"
  "TRIER-C||dense_lamb001_order0|dense_lamb001_softo001|0"
  "TRIER-L|-no_type|notype_dense_lamb001_order0|notype_dense_lamb001_softo001|0"
  "TRIER-S|-no_type|notype_dense_lamb001_order0|notype_dense_lamb001_softo001|1"
  "PACER-LS|-no_type|notype_dense_lamb001_order0|notype_dense_lamb001_softo001|1"
  "PACER-Full||dense_lamb001_order0|dense_lamb001_softo001|1"
)

# For a cell with LOSS_SUF, use it (L_order trained); else use BASE_SUF.
# Cell TRIER=base+no L, TRIER-S=base+no L (S is eval-time only),
# TRIER-C=base+no L, TRIER-L=LOSS+no S, PACER-LS=LOSS+S, PACER-Full=LOSS+S.
# So: use LOSS_SUF only when cell HAS_L; read from BASE_SUF otherwise.
# Actually simpler: map each cell to EXACTLY ONE PT_SUF:
#   TRIER → notype_dense_lamb001_order0
#   TRIER-C → dense_lamb001_order0
#   TRIER-S → notype_dense_lamb001_order0   (S is eval-time -lmd_consec)
#   TRIER-L → notype_dense_lamb001_softo001
#   PACER-LS → notype_dense_lamb001_softo001   (L=training, S=eval)
#   PACER-Full → dense_lamb001_softo001

# NAME|PT_SUF_BASE | PT_SUF_LOSS | TYPE_FLAG | SCORE_PENALTY(LMD_CONSEC)
CELLS2=(
  "TRIER|notype_dense_lamb001_order0|-no_type|0"
  "TRIER-C|dense_lamb001_order0||0"
  "TRIER-L|notype_dense_lamb001_softo001|-no_type|0"
  "TRIER-S|notype_dense_lamb001_order0|-no_type|${LMD_ON}"
  "PACER-LS|notype_dense_lamb001_softo001|-no_type|${LMD_ON}"
  "PACER-Full|dense_lamb001_softo001||${LMD_ON}"
)

get_latest_epoch() {
  ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

# Find all PT dirs matching a suffix (legacy non-suffixed + seedN variants)
find_pt_dirs() {
  local SUF="$1" DIRS=() D
  for D in "save_pt_${SUF}_${VAR}"; do
    [ -d "${D}/model" ] && DIRS+=("$D")
  done
  for S in ${SEEDS}; do
    D="save_pt_${SUF}_${VAR}_seed${S}"
    [ -d "${D}/model" ] && DIRS+=("$D")
  done
  printf '%s\n' "${DIRS[@]}"
}

run_eval_once() {
  # $1=PT_DIR $2=EF $3=EN $4=TYPE_FLAG $5=LMD $6=OUT
  local PT_DIR="$1" EF="$2" EN="$3" TYPE_FLAG="$4" LMD="$5" OUT="$6"
  local LATEST STAGE
  LATEST=$(get_latest_epoch "${PT_DIR}/model")
  [ -z "$LATEST" ] && { echo "    SKIP (no .pth in ${PT_DIR})"; return 1; }
  STAGE="${OUT}.staging"; rm -rf "$STAGE"; mkdir -p "$STAGE"
  ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model" 2>/dev/null
  local TAG=$(basename "$PT_DIR" | sed "s/save_pt_.*_${VAR}_//")
  echo "    eval ${PT_DIR} ep${LATEST} lamb_c=${LMD} -> $(basename "$OUT")"
  CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
      -tf ${DIR}/train-v0.txt -vf ${DIR}/valid-v0.txt -ef "$EF" \
      -vn "$EN" -en "$EN" \
      -cat ${DIR}/${CATE} \
      -n 10728 -n_cat 31 -vec ${VEC} \
      -m test -e ${LATEST} -b 256 \
      ${TYPE_FLAG} -div -lamb ${LAMB} -gamma_consec 0 -lmd_consec ${LMD} -t_mode greedy \
      -start_epoch ${LATEST} -epoch_step 1 \
      -i "${RT_OUT}" -o "$STAGE" 2>&1 | tail -2
  cp "${STAGE}/test_result.txt" "$OUT"
}

echo "############################################################"
echo "# 3-SEED SIX-CELL EVAL  VAR=${VAR}  GPU=${GPU}  seeds=${SEEDS}"
echo "#  RT: ${RT_OUT}"
echo "############################################################"

[ -d "${RT_OUT}/model" ] || { echo "ERROR RT missing"; exit 1; }

for CELL in "${CELLS2[@]}"; do
  IFS='|' read -r NAME PT_SUF TYPE_FLAG LMD <<< "$CELL"
  echo ""
  echo "== ${NAME}  PT_SUF=${PT_SUF}  type=[${TYPE_FLAG:-PACER}]  lamb_c=${LMD}"

  DIRS=$(find_pt_dirs "$PT_SUF")
  if [ -z "$DIRS" ]; then
    echo "    NO PT DIRS FOUND — train_sixcell_3seed.sh 0 first"
    continue
  fi
  echo "    found $(echo "$DIRS" | wc -l) PT dir(s)"

  mkdir -p "${OUTDIR}/${NAME}"
  for PT_DIR in ${DIRS}; do
    TAG=$(basename "$PT_DIR" | sed "s/save_pt_.*_${VAR}//;s/^_//")
    OUTBIG="${OUTDIR}/${NAME}/test_result_big${TAG}.txt"
    run_eval_once "$PT_DIR" "${DIR}/test-v0.txt" "$NEG_BIG" "$TYPE_FLAG" "$LMD" "$OUTBIG"

    if [ -n "$HAS_SMALL" ]; then
      OUTSM="${OUTDIR}/${NAME}/test_result_small${TAG}.txt"
      run_eval_once "$PT_DIR" "./KuaiRec_small_eval/${VAR}/test-v0.txt" "$NEG_SMALL" "$TYPE_FLAG" "$LMD" "$OUTSM"
    fi
  done
done

# ============================================
# AGGREGATE — mean ± std per cell per metric
# ============================================
python3 - <<'PY'
import ast, os, glob, statistics

outdir = "sixcell_firstavg_3seed"
order = ["TRIER", "TRIER-C", "TRIER-L", "TRIER-S", "PACER-LS", "PACER-Full"]
# paper's main metrics + diversity metrics
keys = ["recall@5_f", "recall@10_f", "recall@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ILD@20", "CC@20", "CS@20", "MaxRun@20"]

def gather(cell, tag):
    files = sorted(glob.glob(os.path.join(outdir, cell, f"test_result_{tag}*.txt")))
    rows = []
    for fp in files:
        with open(fp) as f:
            rows.append(ast.literal_eval(f.readline().strip()))
    return rows, files

def agg(rows, k):
    vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    if len(vals) == 0: return None
    return (statistics.mean(vals), statistics.stdev(vals)) if len(vals) >= 2 else (vals[0], 0.0)

for tag in ["small", "big"]:
    print(f"\n{'='*120}")
    print(f"  {tag.upper()} MATRIX — {len(glob.glob(os.path.join(outdir,order[0], f'test_result_{tag}*.txt')))} seeds")
    print(f"{'='*120}")
    width = max(map(len, order))
    header = f"{'cell':<{width}} {'N_seeds':>7} " + "  ".join(f"{k:>18}" for k in keys)
    print(header); print("-"*len(header))
    for name in order:
        rows, files = gather(name, tag)
        n = len(rows)
        row = f"{name:<{width}} {n:>7} "
        for k in keys:
            ms = agg(rows, k)
            if ms is None: row += f"  {'--':>18}"
            else:
                m,s = ms
                row += f"  {m:.4f}±{s:.4f}" if s > 0 else f"  {m:.4f}±   0  "
        print(row)
    print(f"\n  files per cell:")
    for name in order:
        _, files = gather(name, tag)
        print(f"    {name:<{width}}: {[os.path.basename(f) for f in files]}")

# Also save a compact .tex fragment to copy-paste into the paper
for tag in ["small", "big"]:
    tex_path = os.path.join(outdir, "stats", f"mean_std_{tag}.tex")
    with open(tex_path, "w") as f:
        f.write(r"\begin{tabular}{lcccccccccc}" "\n")
        f.write(r"\toprule" "\n")
        f.write(f"& " + " & ".join(k.replace("_f","") for k in keys) + r" \\" "\n")
        f.write(r"\midrule" "\n")
        for name in order:
            rows, _ = gather(name, tag)
            cells = []
            for k in keys:
                ms = agg(rows, k)
                if ms is None: cells.append("--")
                else:
                    m,s = ms
                    cells.append(f"{m:.3f}$\\pm${s:.3f}")
            f.write(f"{name} & " + " & ".join(cells) + r" \\" "\n")
        f.write(r"\bottomrule" "\n")
        f.write(r"\end{tabular}" "\n")
    print(f"\n[LaTeX] -> {tex_path}")

# ─── Answers to the four analysis questions ────────────────────────────────────
print("\n\n" + "="*70)
print("  ANALYSIS QUESTIONS (auto-triage — verify before paper)")
print("="*70)

# 1. Content (TRIER vs TRIER-C, TRIER-L vs PACER-LS): which metric changes most?
def mean_of(cell, k, tag="small"):
    rows, _ = gather(cell, tag)
    ms = agg(rows, k); return ms[0] if ms else None

delta = {}
for metric in ["ndcg@20_f", "recall@20_f", "ILD@20", "CC@20", "CS@20"]:
    a = mean_of("TRIER", metric); b = mean_of("TRIER-C", metric)
    d = mean_of("TRIER-L", metric); c2 = mean_of("PACER-LS", metric)
    if a and b:  # TRIER vs TRIER-C (Content column, no L_order)
        delta.setdefault("Content no-L", {})[metric] = (b - a) / a * 100
    if d and c2:  # TRIER-L vs PACER-LS (Content column, L_order trained)
        delta.setdefault("Content +L", {})[metric] = (c2 - d) / d * 100

print("\n(1) Content contribution — %Δ (PACER - TRIER) / TRIER, positive better:")
for col, per in delta.items():
    best = max(per.items(), key=lambda kv: abs(kv[1]))
    print(f"  [{col}] biggest mover = {best[0]} ({best[1]:+.1f}%)  |  full = {per}")

# 2. L_order vs OrderScore stability: smaller std across seeds → more stable
print("\n(2) L_order vs OrderScore stability (smaller std = more stable):")
for k in keys:
    std_lo = agg(gather("TRIER-L","small")[0], k)
    std_os = agg(gather("TRIER-S","small")[0], k)
    if std_lo and std_os:
        print(f"  {k:<14} L_order std={std_lo[1]:.4f}  OrderScore std={std_os[1]:.4f}  "
              f"{'L_order' if std_lo[1] < std_os[1] else 'OrderScore'} more stable")

# 3. Complementarity: PACER-Full = Content+L_order+S (all ON). Compare to TRIER
print("\n(3) PACER-Full complementarity (Content + L + S all ON vs TRIER base):")
for k in keys:
    a = mean_of("TRIER", k); b = mean_of("PACER-Full", k)
    if a and b:
        sign = "↑" if b > a else "↓"
        print(f"  {k:<14} TRIER={a:.4f}  PACER-Full={b:.4f}  {sign} ({(b-a)/a*100:+.1f}%)")

# 4. Trade-off: accuracy (ndcg) vs repetition (CS, MaxRun)
print("\n(4) Accuracy ↔ repetition trade-off (all cells, small):")
ndcgs = {}; css = {}
for name in order:
    n = mean_of(name, "ndcg@20_f"); c = mean_of(name, "CS@20")
    if n and c: ndcgs[name] = n; css[name] = c
best_acc = max(ndcgs.items(), key=lambda kv: kv[1])
best_rep = min(css.items(), key=lambda kv: kv[1])
print(f"  Best NDCG: {best_acc[0]}={best_acc[1]:.4f}  |  Best (lowest) CS: {best_rep[0]}={best_rep[1]:.4f}")
print(f"  If these are the SAME cell → no trade-off, Full wins outright")
print(f"  If DIFFERENT cells        → trade-off exists; lambda_c (cell S vs L vs Full) is the knob")

PY

echo ""
echo "EVAL DONE -> ${OUTDIR}/"
