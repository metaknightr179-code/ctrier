#!/bin/bash
# =============================================================================
# TRIER Training Pipeline: λ=0.002 and λ=0.003
# Runs BOTH type-embedding and no-type versions in parallel.
#
# Type-embedding models:   save_pt_type_lamb0002_*  / save_pt_type_lamb0003_*
# No-type models:          save_pt_lamb0002_*       / save_pt_lamb0003_*
#
# RT checkpoints:
#   Type:    save_rt_type_<variant>  (reuse existing)
#   No-type: save_rt_kuairec_<variant> (reuse existing)
#
# Usage:
#   bash ~/ctrier/train_pipeline_lamb0002_0003.sh
#   nohup bash ~/ctrier/train_pipeline_lamb0002_0003.sh > ~/ctrier/train_lamb0002_0003.log 2>&1 &
#   nohup bash ~/ctrier_type/train_pipeline_lamb0002_0003.sh > ~/ctrier_type/train_lamb0002_0003.log 2>&1 &
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

# SUFFIX|LAMB|LMD_CONSEC
NEW_CONFIGS=(
    "lamb0002|0.002|0"
    "lamb0003|0.003|0"
)

echo "############################################################"
echo "# Training λ=0.002 and λ=0.003 (type + no-type in parallel)"
echo "# $((${#NEW_CONFIGS[@]} * ${#VARIANTS[@]} * 2)) total runs"
echo "############################################################"
echo ""

# =============================================================================
# STEP 1: Verify RT checkpoints exist
# =============================================================================
echo "[Step 1] Verifying RT checkpoints..."
echo ""

ALL_RT_OK=true
for variant in "${VARIANTS[@]}"; do
    # Type RT
    if [ ! -d "save_rt_type_${variant}/model" ] || [ -z "$(ls save_rt_type_${variant}/model/duorec-*.pth 2>/dev/null)" ]; then
        echo "  WARNING: save_rt_type_${variant} has no checkpoints"
        ALL_RT_OK=false
    fi
    # No-type RT (strip kuairec_ prefix: variant=kuairec_highest_individual -> save_rt_kuairec_highest_individual)
    notype_suffix="${variant#kuairec_}"
    if [ ! -d "save_rt_kuairec_${notype_suffix}/model" ] || [ -z "$(ls save_rt_kuairec_${notype_suffix}/model/duorec-*.pth 2>/dev/null)" ]; then
        echo "  WARNING: save_rt_kuairec_${notype_suffix} has no checkpoints"
        ALL_RT_OK=false
    fi
done

if [ "$ALL_RT_OK" = false ]; then
    echo "Some RT checkpoints missing. Train RT models first."
    echo "  Type:    bash train_pipeline_type.sh (Step 1 only)"
    echo "  No-type: bash train_rt_all.sh"
    exit 1
fi
echo "  All RT checkpoints found."
echo ""

# =============================================================================
# STEP 2: Train PT models (type + no-type in parallel)
# =============================================================================
echo "[Step 2] Training PT models: λ=0.002 and λ=0.003"
echo ""

for config_line in "${NEW_CONFIGS[@]}"; do
    IFS='|' read -r name lamb lmd_consec <<< "$config_line"

    for variant in "${VARIANTS[@]}"; do
        echo "============================================================"
        echo "Training: ${name} / ${variant} (lamb=${lamb})"
        echo "============================================================"

        # ---- TYPE-EMBEDDING VERSION ----
        echo "  [TYPE] Training save_pt_type_${name}_${variant}..."
        type_dir="save_pt_type_${name}_${variant}"
        type_log="pt_type_${name}_${variant}.log"
        rt_type_dir="save_rt_type_${variant}"

        start_epoch=1
        if [ -d "${type_dir}/model" ]; then
            latest=$(ls "${type_dir}/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
            if [ -n "$latest" ]; then
                start_epoch=$(($(basename "$latest" | grep -oE '[0-9]+') + 1))
                echo "    Resuming from epoch ${start_epoch}"
            fi
        fi

        python3 main_pt.py \
            -tf ./KuaiRec_variants/${variant}/train-v0.txt \
            -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
            -ef ./KuaiRec_variants/${variant}/test-v0.txt \
            -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
            -n 10728 -n_cat 31 -e 500 -b 64 -l 5e-4 \
            -div -lamb ${lamb} -lmd_consec ${lmd_consec} \
            -t_mode topk \
            -early_stop -patience 50 -min_delta 0.0001 \
            -start_epoch ${start_epoch} -epoch_step 1 \
            -i ./${rt_type_dir} \
            -o ./${type_dir} \
            2>&1 | tee "$type_log"

        echo "  [TYPE] Done: ${name} / ${variant}"
        echo ""

        # ---- NO-TYPE VERSION ----
        echo "  [NO-TYPE] Training save_pt_${name}_${variant}..."
        notype_dir="save_pt_${name}_${variant}"
        notype_log="pt_${name}_${variant}.log"
        rt_notype_dir="save_rt_kuairec_${variant#kuairec_}"

        start_epoch=1
        if [ -d "${notype_dir}/model" ]; then
            latest=$(ls "${notype_dir}/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
            if [ -n "$latest" ]; then
                start_epoch=$(($(basename "$latest" | grep -oE '[0-9]+') + 1))
                echo "    Resuming from epoch ${start_epoch}"
            fi
        fi

        python3 main_pt.py \
            -tf ./KuaiRec_variants/${variant}/train-v0.txt \
            -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
            -ef ./KuaiRec_variants/${variant}/test-v0.txt \
            -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
            -n 10728 -n_cat 31 -e 500 -b 64 -l 5e-4 \
            -div -lamb ${lamb} -lmd_consec ${lmd_consec} \
            -t_mode topk \
            -early_stop -patience 50 -min_delta 0.0001 \
            -start_epoch ${start_epoch} -epoch_step 1 \
            -i ./${rt_notype_dir} \
            -o ./${notype_dir} \
            2>&1 | tee "$notype_log"

        echo "  [NO-TYPE] Done: ${name} / ${variant}"
        echo ""
    done
done

echo ""
echo "############################################################"
echo "Training complete!"
echo "############################################################"
echo ""
echo "Type results:    save_pt_type_lamb0002_*/test_result.txt"
echo "                 save_pt_type_lamb0003_*/test_result.txt"
echo "No-type results: save_pt_lamb0002_*/test_result.txt"
echo "                 save_pt_lamb0003_*/test_result.txt"
echo ""
echo "Next: bash eval_lamb0002_0003.sh"
