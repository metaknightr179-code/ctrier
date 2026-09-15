#!/bin/bash
# =============================================================================
# MASTER EVAL RUNNER — chains all eval scripts sequentially on ONE GPU,
# with nohup so you can disconnect from AutoDL without killing anything.
#
# Each eval script runs in the foreground (so they run one at a time on GPU 0,
# avoiding OOM), but the *entire* master script is launched via nohup, so
# closing SSH doesn't kill the whole pipeline.
#
# Log layout (all under ./eval_logs/):
#   run_all.log       — master log (timestamps + per-eval status)
#   sixcell.log       — eval_sixcell_3seed.sh output
#   singletons.log    — eval_singletons.sh output
#   newds_ML1M.log    — eval_newds.sh ML1M output
#   newds_KuaiRand1K.log
#   newds_MicroLens.log
#   penalty_sweep.log — eval_penalty_sweep_firstavg.sh output
#   temp_sweep.log    — eval_temp_sweep_firstavg.sh output
#   consecgamma.log   — eval_consecgamma_grid_firstavg.sh output
#
# Usage:
#   cd ~/ctrier
#   pkill -KILL -f main_pt; pkill -KILL -f train_; sleep 2
#   nohup bash run_all_evals.sh > eval_logs/run_all.log 2>&1 &
#
# Check progress:
#   tail -f eval_logs/run_all.log        # master
#   tail -f eval_logs/<script>.log       # specific eval
#   nvidia-smi                           # GPU usage
#
# Resume (if it crashes mid-way): just re-run the same nohup command above.
# eval_newds.sh has FORCE_REEVAL set, so it re-evaluates even if results exist.
# All other eval scripts either have no skip guard (eval_singletons.sh) or
# will re-run on the NEXT epoch sweep naturally.
# =============================================================================

set -u
cd "$(dirname "$0")"

mkdir -p eval_logs
export CUDA_VISIBLE_DEVICES=0
FORCE_REEVAL=1
export FORCE_REEVAL

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a eval_logs/run_all.log; }
run_eval() {
  local TAG="$1"; shift
  log "START  ${TAG}  → eval_logs/${TAG}.log"
  bash "$@" > "eval_logs/${TAG}.log" 2>&1
  local RC=$?
  if [ $RC -eq 0 ]; then
    log "DONE   ${TAG}"
  else
    log "ERROR  ${TAG}  exit_code=${RC}"
  fi
  return $RC
}

log "=============================================="
log "MASTER EVAL RUNNER STARTED  GPU=${CUDA_VISIBLE_DEVICES}"
log "=============================================="

# ── 1. Six-cell ablation (3-seed) ────────────────────────────
run_eval sixcell   eval_sixcell_3seed.sh

# ── 2. Singletons (γ=1 + all 4 KuaiRec variants + 3 new datasets) ─
run_eval singletons  eval_singletons.sh

# ── 3. New datasets standalone (ML1M, KuaiRand1K, MicroLens) ──
run_eval newds_ML1M         eval_newds.sh 0 ML1M
run_eval newds_KuaiRand1K   eval_newds.sh 0 KuaiRand1K
run_eval newds_MicroLens    eval_newds.sh 0 MicroLens

# ── 4. Sweeps (all on KuaiRec First-Average, small-matrix) ──
run_eval penalty_sweep   eval_penalty_sweep_firstavg.sh
run_eval temp_sweep      eval_temp_sweep_firstavg.sh
run_eval consecgamma     eval_consecgamma_grid_firstavg.sh

log "=============================================="
log "MASTER EVAL RUNNER COMPLETE"
log "=============================================="
echo ""
echo "Logs are in ./eval_logs/"
echo "  tail -f eval_logs/run_all.log       # master progress"
echo "  tail -f eval_logs/<script>.log      # specific eval detail"
echo ""
echo "Now run analyze_results.py:"
echo "  python3 analyze_results.py --sixcell --proto small"
echo "  python3 analyze_results.py --singletons --proto small"
echo "  python3 analyze_results.py --singletons --proto big"
echo "  python3 analyze_results.py --gamma_sweep --proto small"
echo "  python3 analyze_results.py --temp_sweep --proto small"
echo "  python3 analyze_results.py --penalty_sweep --proto small"
