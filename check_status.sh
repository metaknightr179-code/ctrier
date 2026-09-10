#!/bin/bash
# Quick status check for all training pipelines (baselines + TRIER RT/PT).
# Usage: bash check_status.sh
# Prints: per-dir epoch count, DONE marker, early-stopped, or not started.
# =============================================================================
cd "$(dirname "$0")"
set -u

MAX_EPOCHS=1000
BASELINE_EPOCHS=500

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

CONFIGS=(nodiv lamb0002 lamb0005 lamb0005_consec0001 lamb001 lamb005 lamb01)

# ---- Baselines (SASRec, GRU4Rec, BERT4Rec) ----
echo "============================================================"
echo " BASELINES (target: ${BASELINE_EPOCHS} epochs or early-stopped)"
echo "============================================================"
for model in sasrec gru4rec bert4rec; do
    for VAR in "${VARIANTS[@]}"; do
        DIR="save_${model}_${VAR}"
        LOG="train_${model}_${VAR}.log"
        DONE="${DIR}/DONE"
        CKPT="${DIR}/${model}_best.pth"

        UPPER=$(echo "$model" | tr '[:lower:]' '[:upper:]')
        if [ -f "$DONE" ]; then
            # Check for early stop marker
            if grep -q "Early stopping" "$LOG" 2>/dev/null; then
                echo "  ${UPPER} ${VAR}: DONE (early-stopped)"
            else
                echo "  ${UPPER} ${VAR}: DONE"
            fi
        elif [ -f "$CKPT" ]; then
            echo "  ${UPPER} ${VAR}: COMPLETE (checkpoint exists)"
        elif [ -f "$LOG" ]; then
            # Count epoch lines in log
            EP=$(grep -c "^Epoch " "$LOG" 2>/dev/null)
            EP=${EP:-0}
            if [ "$EP" -gt 0 ]; then
                echo "  ${UPPER} ${VAR}: training (${EP}/${BASELINE_EPOCHS} epochs)"
            else
                echo "  ${UPPER} ${VAR}: started (no epochs logged yet)"
            fi
        else
            echo "  ${UPPER} ${VAR}: not started"
        fi
    done
done

# ---- RT checkpoints ----
echo ""
echo "============================================================"
echo " RT (target: ${MAX_EPOCHS} epochs)"
echo "============================================================"
for VAR in "${VARIANTS[@]}"; do
    DIR="save_rt_fix_${VAR}"
    LOG="${DIR}/train_result.txt"
    DONE="${DIR}/DONE"

    if [ -f "$DONE" ]; then
        echo "  RT ${VAR}: DONE"
    elif [ -f "$LOG" ]; then
        LINES=$(wc -l < "$LOG" 2>/dev/null); LINES=${LINES:-0}
        if [ "$LINES" -ge "$MAX_EPOCHS" ]; then
            echo "  RT ${VAR}: ${MAX_EPOCHS} epochs (complete)"
        else
            echo "  RT ${VAR}: training (${LINES}/${MAX_EPOCHS} epochs)"
        fi
    else
        echo "  RT ${VAR}: not started"
    fi
done

# ---- PT (non-dense fixrt) ----
echo ""
echo "============================================================"
echo " PT NON-DENSE FIXRT (target: ${MAX_EPOCHS} epochs or early-stopped)"
echo "============================================================"
for FAM in "type|save_pt_fixrt_|" "notype|save_pt_notype_fixrt_|-no_type"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX FLAG <<< "$FAM"
    for CFG in "${CONFIGS[@]}"; do
        for VAR in "${VARIANTS[@]}"; do
            DIR="${DIR_PREFIX}${CFG}_${VAR}"
            LOG="${DIR}/train_result.txt"
            DONE="${DIR}/DONE"

            if [ -f "$DONE" ]; then
                echo "  ${FAM_NAME} ${CFG} ${VAR}: DONE"
            elif [ -f "$LOG" ]; then
                LINES=$(wc -l < "$LOG" 2>/dev/null); LINES=${LINES:-0}
                if [ "$LINES" -ge "$MAX_EPOCHS" ]; then
                    echo "  ${FAM_NAME} ${CFG} ${VAR}: ${MAX_EPOCHS} epochs (complete)"
                else
                    echo "  ${FAM_NAME} ${CFG} ${VAR}: partial (${LINES}/${MAX_EPOCHS})"
                fi
            elif [ -d "${DIR}/model" ]; then
                echo "  ${FAM_NAME} ${CFG} ${VAR}: started (no log)"
            else
                : #  don't print not-started to reduce noise
            fi
        done
    done
done

# ---- PT (dense) ----
echo ""
echo "============================================================"
echo " PT DENSE (target: ${MAX_EPOCHS} epochs or early-stopped)"
echo "============================================================"
for FAM in "type|save_pt_dense_|" "notype|save_pt_notype_dense_|-no_type"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX FLAG <<< "$FAM"
    for CFG in "${CONFIGS[@]}"; do
        for VAR in "${VARIANTS[@]}"; do
            DIR="${DIR_PREFIX}${CFG}_${VAR}"
            LOG="${DIR}/train_result.txt"
            DONE="${DIR}/DONE"

            if [ -f "$DONE" ]; then
                echo "  ${FAM_NAME} ${CFG} ${VAR}: DONE"
            elif [ -f "$LOG" ]; then
                LINES=$(wc -l < "$LOG" 2>/dev/null); LINES=${LINES:-0}
                if [ "$LINES" -ge "$MAX_EPOCHS" ]; then
                    echo "  ${FAM_NAME} ${CFG} ${VAR}: ${MAX_EPOCHS} epochs (complete)"
                else
                    echo "  ${FAM_NAME} ${CFG} ${VAR}: partial (${LINES}/${MAX_EPOCHS})"
                fi
            elif [ -d "${DIR}/model" ]; then
                echo "  ${FAM_NAME} ${CFG} ${VAR}: started (no log)"
            fi
        done
    done
done

# ---- PT (author families) ----
echo ""
echo "============================================================"
echo " PT AUTHOR FAMILIES (target: ${MAX_EPOCHS} epochs or early-stopped)"
echo "============================================================"
for FAM in "author|save_pt_author_fixrt_|" "typeauthor|save_pt_typeauthor_fixrt_|"; do
    IFS='|' read -r FAM_NAME DIR_PREFIX FLAG <<< "$FAM"
    for CFG in "${CONFIGS[@]}"; do
        for VAR in "${VARIANTS[@]}"; do
            DIR="${DIR_PREFIX}${CFG}_${VAR}"
            LOG="${DIR}/train_result.txt"
            DONE="${DIR}/DONE"

            if [ -f "$DONE" ]; then
                echo "  ${FAM_NAME} ${CFG} ${VAR}: DONE"
            elif [ -f "$LOG" ]; then
                LINES=$(wc -l < "$LOG" 2>/dev/null); LINES=${LINES:-0}
                if [ "$LINES" -ge "$MAX_EPOCHS" ]; then
                    echo "  ${FAM_NAME} ${CFG} ${VAR}: ${MAX_EPOCHS} epochs (complete)"
                else
                    echo "  ${FAM_NAME} ${CFG} ${VAR}: partial (${LINES}/${MAX_EPOCHS})"
                fi
            elif [ -d "${DIR}/model" ]; then
                echo "  ${FAM_NAME} ${CFG} ${VAR}: started (no log)"
            fi
        done
    done
done

# ---- Summary counts ----
echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"
# Baselines
b_done=0; b_partial=0; b_not=0
for model in sasrec gru4rec bert4rec; do
    for VAR in "${VARIANTS[@]}"; do
        DIR="save_${model}_${VAR}"
        LOG="train_${model}_${VAR}.log"
        CKPT="${DIR}/${model}_best.pth"
        DONE="${DIR}/DONE"
        if [ -f "$DONE" ] || [ -f "$CKPT" ]; then b_done=$((b_done+1))
        elif [ -f "$LOG" ]; then b_partial=$((b_partial+1))
        else b_not=$((b_not+1)); fi
    done
done
echo "  Baselines: ${b_done} done / ${b_partial} training / ${b_not} not started (of $(( 3*4 )))"

# RT
rt_done=0; rt_partial=0; rt_not=0
for VAR in "${VARIANTS[@]}"; do
    DIR="save_rt_fix_${VAR}"
    LOG="${DIR}/train_result.txt"
    DONE="${DIR}/DONE"
    if [ -f "$DONE" ]; then rt_done=$((rt_done+1))
    elif [ -f "$LOG" ]; then
        LINES=$(wc -l < "$LOG" 2>/dev/null); LINES=${LINES:-0}
        [ "$LINES" -ge "$MAX_EPOCHS" ] && rt_done=$((rt_done+1)) || rt_partial=$((rt_partial+1))
    else rt_not=$((rt_not+1)); fi
done
echo "  RT: ${rt_done} done / ${rt_partial} partial / ${rt_not} not started (of 4)"

# PT all families
for LABEL in "fixrt:save_pt_fixrt_:save_pt_notype_fixrt_" \
             "dense:save_pt_dense_:save_pt_notype_dense_" \
             "author:save_pt_author_fixrt_:save_pt_typeauthor_fixrt_"; do
    IFS=':' read -r PNAME P1 P2 <<< "$LABEL"
    pt_done=0; pt_partial=0; pt_not=0
    for DIR_PREFIX in "$P1" "$P2"; do
        for CFG in "${CONFIGS[@]}"; do
            for VAR in "${VARIANTS[@]}"; do
                DIR="${DIR_PREFIX}${CFG}_${VAR}"
                LOG="${DIR}/train_result.txt"
                DONE="${DIR}/DONE"
                if [ -f "$DONE" ]; then pt_done=$((pt_done+1))
                elif [ -f "$LOG" ]; then
                    LINES=$(wc -l < "$LOG" 2>/dev/null); LINES=${LINES:-0}
                    [ "$LINES" -ge "$MAX_EPOCHS" ] && pt_done=$((pt_done+1)) || pt_partial=$((pt_partial+1))
                elif [ -d "${DIR}/model" ]; then pt_partial=$((pt_partial+1))
                else pt_not=$((pt_not+1)); fi
            done
        done
    done
    total=$(( 2*7*4 ))
    echo "  PT ${PNAME}: ${pt_done} done / ${pt_partial} partial / ${pt_not} not started (of ${total})"
done
