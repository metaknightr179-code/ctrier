#!/bin/bash
# =============================================================================
# EVAL ALL SINGLETON CHECKPOINTS — one command, one consolidated table.
#
# Covers every "standard-settings" checkpoint that wasn't part of a grid sweep:
#   Group A — γ=1.0 single-point sweep (KuaiRec First-Average)
#   Group B — γ=0 best config on all 4 KuaiRec variants
#   Group C — γ=0 best config on NEW DATASETS (ML1M, KuaiRand1K, MicroLens)
#
# Pipeline per checkpoint: greedy diverse decode  (-div -lamb 0.01 -gamma_consec 0
# -t_mode greedy), lambda_c=0 in scorer (same canonical setup as every other sweep).
#
# Outputs: one result file per checkpoint directory (big + small if available).
# Final table printed at the end.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash eval_singletons.sh
# =============================================================================
set -u
GPU=${CUDA_VISIBLE_DEVICES:-0}

# ──────────────────────────────────────────────────────────────
# REGISTRY — one row per singleton checkpoint (all fields pipe-separated)
#   PT_DIR | DISPLAY | GAMMA_LABEL | N_ITEMS | N_CATEGORIES | RT_DIR |
#   DATA_DIR | CATE_FILE | VEC_FILE | NEG_FILE | TYPE_FLAG (-no_type or empty)
# ──────────────────────────────────────────────────────────────
SINGLES=(
  # Group A: γ=1.0 (KuaiRec First-Average, type family — the single run user asked for)
  "save_pt_dense_lamb001_softo1_kuairec_first_average|KuaiRec-FirstAvg|γ=1.0|10728|31|save_rt_fix_kuairec_first_average|KuaiRec_variants/kuairec_first_average|kuairec_cate.txt|KuaiRec_variants/kuairec_vec.npy|KuaiRec-random-sample_size=99-seed=4444.txt|"

  # Group B: γ=0 best config on ALL 4 KuaiRec variants
  "save_pt_dense_lamb001_order0_kuairec_first_average|KuaiRec-FirstAvg|γ=0|10728|31|save_rt_fix_kuairec_first_average|KuaiRec_variants/kuairec_first_average|kuairec_cate.txt|KuaiRec_variants/kuairec_vec.npy|KuaiRec-random-sample_size=99-seed=4444.txt|"
  "save_pt_dense_lamb001_order0_kuairec_first_individual|KuaiRec-FirstIndv|γ=0|10728|31|save_rt_fix_kuairec_first_individual|KuaiRec_variants/kuairec_first_individual|kuairec_cate.txt|KuaiRec_variants/kuairec_vec.npy|KuaiRec-random-sample_size=99-seed=4444.txt|"
  "save_pt_dense_lamb001_order0_kuairec_highest_average|KuaiRec-HighAvg|γ=0|10728|31|save_rt_fix_kuairec_highest_average|KuaiRec_variants/kuairec_highest_average|kuairec_cate.txt|KuaiRec_variants/kuairec_vec.npy|KuaiRec-random-sample_size=99-seed=4444.txt|"
  "save_pt_dense_lamb001_order0_kuairec_highest_individual|KuaiRec-HighIndv|γ=0|10728|31|save_rt_fix_kuairec_highest_individual|KuaiRec_variants/kuairec_highest_individual|kuairec_cate.txt|KuaiRec_variants/kuairec_vec.npy|KuaiRec-random-sample_size=99-seed=4444.txt|"

  # Group C: γ=0 best config on NEW DATASETS
  "save_pt_dense_lamb001_order0_ML1M|ML-1M|γ=0|3126|18|save_rt_fix_ML1M|ML1M|ml1m_cate.txt|ML1M/ml1m_vec.npy|ML1M-random-sample_size=99-seed=4444.txt|"
  "save_pt_dense_lamb001_order0_KuaiRand1K|KuaiRand|γ=0|20001|44|save_rt_fix_KuaiRand1K|KuaiRand1K|kuairand_cate.txt|KuaiRand1K/kuairand_vec.npy|KuaiRand-random-sample_size=99-seed=4444.txt|"
  "save_pt_dense_lamb001_order0_MicroLens|MicroLens|γ=0|26923|57|save_rt_fix_MicroLens|MicroLens|microlens_cate.txt|MicroLens/microlens_vec.npy|MicroLens-random-sample_size=99-seed=4444.txt|"
)

echo "############################################################"
echo "SINGLETON CHECKPOINT EVAL — GPU=${GPU}"
echo "Registered: ${#SINGLES[@]} PT dirs"
echo "############################################################"

get_latest_epoch() {
  ls "${1}"/model/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

for ROW in "${SINGLES[@]}"; do
  IFS='|' read -r PT_DIR DISPLAY GAMMA N NCAT RT_DIR DIR CATE VEC NEG TYPE_FLAG <<< "$ROW"

  echo ""
  echo "== ${DISPLAY} (${GAMMA}) = ${PT_DIR}"

  # Guards
  if [ ! -d "${PT_DIR}/model" ]; then
    echo "  SKIP — PT dir missing"
    continue
  fi
  LATEST=$(get_latest_epoch "${PT_DIR}")
  [ -z "$LATEST" ] && { echo "  SKIP — no .pth"; continue; }
  if [ ! -d "${RT_DIR}/model" ]; then
    echo "  SKIP — RT checkpoint missing: ${RT_DIR}"
    continue
  fi

  STAGE="./save_denseeval_staging/single_${PT_DIR}"
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"

  # ── BIG matrix ──────────────────────────────────────────────
  echo "  → big-matrix  pt=ep${LATEST}"
  CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
      -tf "${DIR}/train-v0.txt" -vf "${DIR}/valid-v0.txt" -ef "${DIR}/test-v0.txt" \
      -vn "${DIR}/${NEG}" -en "${DIR}/${NEG}" \
      -cat "${DIR}/${CATE}" -vec "${VEC}" \
      -n ${N} -n_cat ${NCAT} \
      ${TYPE_FLAG} \
      -m test -e ${LATEST} -b 256 \
      -div -lamb 0.01 -gamma_consec 0 -t_mode greedy \
      -start_epoch ${LATEST} -epoch_step 1 \
      -i "${RT_DIR}" -o "${STAGE}" 2>&1 | tail -1
  cp "${STAGE}/test_result.txt" "${PT_DIR}/test_result_gridorder.txt"

  # ── SMALL matrix — only for KuaiRec variants ────────────────
  SMALL_DIR="./KuaiRec_small_eval/${DISPLAY}"
  if [ -f "${SMALL_DIR}/test-v0.txt" ] && [ -f "${SMALL_DIR}/${NEG}" ]; then
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    ln -s "$(cd "${PT_DIR}/model" && pwd)" "$STAGE/model"
    echo "  → small-matrix  pt=ep${LATEST}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf "${DIR}/train-v0.txt" -vf "${DIR}/valid-v0.txt" -ef "${SMALL_DIR}/test-v0.txt" \
        -vn "${SMALL_DIR}/${NEG}" -en "${SMALL_DIR}/${NEG}" \
        -cat "${DIR}/${CATE}" -vec "${VEC}" \
        -n ${N} -n_cat ${NCAT} \
        ${TYPE_FLAG} \
        -m test -e ${LATEST} -b 256 \
        -div -lamb 0.01 -gamma_consec 0 -t_mode greedy \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i "${RT_DIR}" -o "${STAGE}" 2>&1 | tail -1
    cp "${STAGE}/test_result.txt" "${PT_DIR}/test_result_small_gridorder.txt"
  fi
done

# ──────────────────────────────────────────────────────────────
# CONSOLIDATED TABLE
# ──────────────────────────────────────────────────────────────
echo ""
echo "############################################################"
echo "CONSOLIDATED RESULTS (big-matrix protocol)"
echo "############################################################"
python3 - <<'PY'
import os, ast, glob

PT_DIRS = [
    ("save_pt_dense_lamb001_softo1_kuairec_first_average",      "KuaiRec-FirstAvg",  "γ=1.0"),
    ("save_pt_dense_lamb001_order0_kuairec_first_average",      "KuaiRec-FirstAvg",  "γ=0"),
    ("save_pt_dense_lamb001_order0_kuairec_first_individual",   "KuaiRec-FirstIndv", "γ=0"),
    ("save_pt_dense_lamb001_order0_kuairec_highest_average",   "KuaiRec-HighAvg",   "γ=0"),
    ("save_pt_dense_lamb001_order0_kuairec_highest_individual", "KuaiRec-HighIndv",  "γ=0"),
    ("save_pt_dense_lamb001_order0_ML1M",                      "ML-1M",             "γ=0"),
    ("save_pt_dense_lamb001_order0_KuaiRand1K",                "KuaiRand",          "γ=0"),
    ("save_pt_dense_lamb001_order0_MicroLens",                 "MicroLens",         "γ=0"),
]

keys = ["recall@5_f","recall@10_f","recall@20_f",
        "ndcg@5_f","ndcg@10_f","ndcg@20_f",
        "ILD@20","CC@20","CS@20","MaxRun@20"]
hdr = f"{'variant':<20} {'γ':>5}  " + " ".join(f"{k:>11}" for k in keys)
print(hdr); print("-"*len(hdr))

for PT_DIR, VAR, GAM in PT_DIRS:
    fp_big = f"{PT_DIR}/test_result_gridorder.txt"
    fp_legacy = f"{PT_DIR}/test_result.txt"
    fp = fp_big if os.path.exists(fp_big) else fp_legacy
    if not os.path.exists(fp):
        print(f"{VAR:<20} {GAM:>5}  NO_RESULT  ({PT_DIR})")
        continue
    with open(fp) as f:
        d = ast.literal_eval(f.readline().strip())
    row = f"{VAR:<20} {GAM:>5}  "
    row += " ".join(f"{d.get(k, float('nan')):>11.4f}" for k in keys)
    print(row)

print("\n" + "="*len(hdr))
print("SMALL-MATRIX (KuaiRec variants only)")
print("="*len(hdr))
for PT_DIR, VAR, GAM in PT_DIRS:
    if "KuaiRec" not in VAR: continue
    fp = f"{PT_DIR}/test_result_small_gridorder.txt"
    if not os.path.exists(fp):
        print(f"{VAR:<20} {GAM:>5}  NO_RESULT")
        continue
    with open(fp) as f:
        d = ast.literal_eval(f.readline().strip())
    row = f"{VAR:<20} {GAM:>5}  "
    row += " ".join(f"{d.get(k, float('nan')):>11.4f}" for k in keys)
    print(row)
PY

rm -rf ./save_denseeval_staging
echo ""
echo "############################################################"
echo "DONE — results in each PT_DIR/test_result*.txt"
echo "############################################################"
