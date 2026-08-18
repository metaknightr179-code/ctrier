#!/bin/bash
# =============================================================================
# Disk space cleanup script for remote instance
# Frees space by deleting intermediate checkpoints and large logs
# SAFE TO RUN: only deletes files that can be regenerated or are non-critical
# =============================================================================

set -e

cd /root/ctrier

echo "=== DISK USAGE BEFORE CLEANUP ==="
df -h | grep -E "File|/root|/$"
echo ""
du -sh save_* tensorboard_logs 2>/dev/null || echo "No directories found"
echo ""

TOTAL_FREED=0

# ------------------------------------------------------------------
# 1. Delete TensorBoard logs (largest, not needed after training)
# ------------------------------------------------------------------
if [ -d "./tensorboard_logs" ]; then
    SIZE=$(du -sb ./tensorboard_logs | cut -f1)
    rm -rf ./tensorboard_logs
    MB=$((SIZE / 1024 / 1024))
    TOTAL_FREED=$((TOTAL_FREED + SIZE))
    echo "[1/4] Deleted tensorboard_logs: freed ${MB} MB"
else
    echo "[1/4] tensorboard_logs not found — skipping"
fi

# ------------------------------------------------------------------
# 2. Keep ONLY the latest checkpoint per model dir (delete all older epochs)
# ------------------------------------------------------------------
echo "[2/4] Cleaning intermediate checkpoints (keeping latest only per model dir)..."
CKPTS_FREED=0
for dir in save_*/model; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    NUM_BEFORE=$(ls "$dir/"duorec-*.pth 2>/dev/null | wc -l)
    if [ "$NUM_BEFORE" -le 1 ]; then
        continue
    fi
    # Calculate size of files to be deleted
    TO_DELETE=$(ls -t "$dir/"duorec-*.pth 2>/dev/null | tail -n +2)
    if [ -n "$TO_DELETE" ]; then
        SIZE=$(du -c $TO_DELETE 2>/dev/null | tail -1 | cut -f1)
        CKPTS_FREED=$((CKPTS_FREED + SIZE))
        ls -t "$dir/"duorec-*.pth 2>/dev/null | tail -n +2 | xargs -r rm -f
    fi
    NUM_AFTER=$(ls "$dir/"duorec-*.pth 2>/dev/null | wc -l)
    echo "  $dir: $NUM_BEFORE -> $NUM_AFTER checkpoints"
done
MB=$((CKPTS_FREED / 1024 / 1024))
TOTAL_FREED=$((TOTAL_FREED + CKPTS_FREED))
echo "  Freed ~${MB} MB from intermediate checkpoints"

# ------------------------------------------------------------------
# 3. Delete old training/compression logs (keep recent ones)
# ------------------------------------------------------------------
echo "[3/4] Cleaning old log files (> 7 days)..."
if ls *.log 1>/dev/null 2>&1; then
    SIZE=$(find . -maxdepth 1 -name "*.log" -mtime +7 -exec du -cb {} + 2>/dev/null | tail -1 | cut -f1)
    if [ -n "$SIZE" ] && [ "$SIZE" -gt 0 ]; then
        find . -maxdepth 1 -name "*.log" -mtime +7 -delete
        MB=$((SIZE / 1024 / 1024))
        TOTAL_FREED=$((TOTAL_FREED + SIZE))
        echo "  Deleted old logs: freed ${MB} MB"
    else
        echo "  No old logs to delete"
    fi
fi

# Also delete nohup.out if large
if [ -f "nohup.out" ]; then
    SIZE=$(stat -c%s nohup.out 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 104857600 ]; then  # > 100MB
        MB=$((SIZE / 1024 / 1024))
        rm -f nohup.out
        TOTAL_FREED=$((TOTAL_FREED + SIZE))
        echo "  Deleted large nohup.out (${MB} MB)"
    fi
fi

# ------------------------------------------------------------------
# 4. Clear pip/pip cache if large
# ------------------------------------------------------------------
echo "[4/4] Cleaning caches..."
PIP_CACHE=$(pip cache dir 2>/dev/null || echo "")
if [ -n "$PIP_CACHE" ] && [ -d "$PIP_CACHE" ]; then
    SIZE=$(du -sb "$PIP_CACHE" 2>/dev/null | cut -f1)
    if [ "$SIZE" -gt 104857600 ]; then  # > 100MB
        pip cache purge 2>/dev/null || true
        MB=$((SIZE / 1024 / 1024))
        TOTAL_FREED=$((TOTAL_FREED + SIZE))
        echo "  Purged pip cache: freed ${MB} MB"
    else
        echo "  Pip cache small (<100MB), keeping"
    fi
fi

# Clear apt cache
if command -v apt-get &> /dev/null; then
    SIZE=$(du -sb /var/cache/apt/archives 2>/dev/null | cut -f1)
    if [ -n "$SIZE" ] && [ "$SIZE" -gt 10485760 ]; then  # > 10MB
        apt-get clean 2>/dev/null || true
        MB=$((SIZE / 1024 / 1024))
        TOTAL_FREED=$((TOTAL_FREED + SIZE))
        echo "  Cleaned apt cache: freed ${MB} MB"
    fi
fi

echo ""
echo "=== DISK USAGE AFTER CLEANUP ==="
df -h | grep -E "File|/root|/$"
echo ""
TOTAL_MB=$((TOTAL_FREED / 1024 / 1024))
echo "=== TOTAL FREED: ~${TOTAL_MB} MB ($((TOTAL_MB / 1024)) GB) ==="
echo ""
echo "Checkpoints retained: latest epoch only per model"
echo "To find best epochs, run: ./eval_best.sh  (but requires checkpoints)"
echo "To clean EVEN MORE aggressively, edit this script to:"
echo "  - Delete entire save_* directories (after downloading checkpoints locally)"
echo "  - Delete KuaiRec_variants/ (raw data, re-downloadable)"
