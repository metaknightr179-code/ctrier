#!/bin/bash
# =============================================================================
# Master training pipeline: trains RT models, then PT models, sequentially.
#
# Pipeline:
#   1. train_rt_all.sh   (RT models for all 4 variants)
#   2. train_all.sh      (PT models: nodiv, lamb0005, lamb001, lamb005, lamb01, consec)
#
# RT must complete before PT starts (PT depends on RT checkpoints).
# This script blocks until RT finishes, then automatically starts PT.
#
# Usage:
#   bash train_pipeline.sh                       # run full pipeline
#   nohup bash train_pipeline.sh > train_pipeline.log 2>&1 &   # background
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "############################################################"
echo "# TRIER Training Pipeline"
echo "# Step 1: RT training (train_rt_all.sh)"
echo "# Step 2: PT training (train_all.sh)"
echo "############################################################"
echo ""

# -----------------------------
# Step 1: Train RT models
# -----------------------------
echo "============================================================"
echo "[STEP 1/2] Training RT models..."
echo "============================================================"
bash train_rt_all.sh
RT_EXIT=$?
if [ "$RT_EXIT" -ne 0 ]; then
    echo "ERROR: RT training failed (exit code ${RT_EXIT}). Aborting pipeline."
    exit "$RT_EXIT"
fi
echo ""
echo "============================================================"
echo "[STEP 1/2] RT training complete."
echo "============================================================"
echo ""

# -----------------------------
# Step 2: Train PT models
# -----------------------------
echo "============================================================"
echo "[STEP 2/2] Training PT models..."
echo "============================================================"
bash train_all.sh
PT_EXIT=$?
if [ "$PT_EXIT" -ne 0 ]; then
    echo "WARNING: PT training exited with code ${PT_EXIT}. Check pt_*.log files."
fi
echo ""

# -----------------------------
# Summary
# -----------------------------
echo "############################################################"
echo "# Pipeline Complete"
echo "#   RT exit code: ${RT_EXIT}"
echo "#   PT exit code: ${PT_EXIT}"
echo "############################################################"
echo ""
echo "Next step: bash eval_all.sh"
