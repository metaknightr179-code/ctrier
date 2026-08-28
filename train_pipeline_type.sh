#!/bin/bash
# =============================================================================
# TRIER Type Embedding Pipeline
# Purpose: Train RT then PT models with RecFormer-style type embeddings
# Step 1: Train RT models (batch_size=64, 500 epochs, early stopping)
# Step 2: Train PT models with type embeddings (6 configs x 4 variants)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

CONFIGS=(
    "nodiv|0|0"
    "lamb0005|0.005|0"
    "lamb001|0.01|0"
    "lamb005|0.05|0"
    "lamb01|0.1|0"
    "consec0001|0.01|0.01"
)

echo "============================================"
echo "TRIER Type Embedding Pipeline"
echo "Step 1: Train RT models (batch_size=64)"
echo "Step 2: Train PT models with type embeddings"
echo "============================================"

# =============================================================================
# STEP 1: Train RT models
# =============================================================================
echo ""
echo "[Step 1] Training RT models..."
echo ""

for variant in "${VARIANTS[@]}"; do
    rt_dir="save_rt_type_${variant}"
    log_file="rt_type_${variant}.log"

    # Skip if already trained (resume support)
    if [ "$1" == "-r" ] && [ -d "${rt_dir}/model" ]; then
        latest=$(ls "${rt_dir}/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
        if [ -n "$latest" ]; then
            echo "  SKIPPED: ${variant} (already trained, checkpoint: $(basename $latest))"
            continue
        fi
    fi

    echo "Training RT: ${variant}"

    # Get resume epoch
    start_epoch=1
    if [ -d "${rt_dir}/model" ]; then
        latest=$(ls "${rt_dir}/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
        if [ -n "$latest" ]; then
            start_epoch=$(($(basename "$latest" | grep -oE '[0-9]+') + 1))
            echo "  Resuming from epoch ${start_epoch}"
        fi
    fi

    python3 main_rt.py \
        -tf ./KuaiRec_variants/${variant}/train-v0.txt \
        -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
        -ef ./KuaiRec_variants/${variant}/test-v0.txt \
        -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -e 500 -b 64 -l 5e-4 \
        -early_stop -patience 50 -min_delta 0.0001 \
        -start_epoch ${start_epoch} -epoch_step 1 \
        -o ./${rt_dir} \
        2>&1 | tee "$log_file"

    echo "  Done: RT ${variant}"
    echo ""
done

# =============================================================================
# STEP 2: Train PT models with type embeddings
# =============================================================================
echo ""
echo "[Step 2] Training PT models with type embeddings..."
echo "Configs: ${#CONFIGS[@]} x Variants: ${#VARIANTS[@]} = $((${#CONFIGS[@]} * ${#VARIANTS[@]})) runs"
echo ""

for config_line in "${CONFIGS[@]}"; do
    IFS='|' read -r name lamb lmd_consec <<< "$config_line"

    for variant in "${VARIANTS[@]}"; do
        echo "----------------------------------------"
        echo "Training PT (type): ${name} / ${variant}"
        echo "  lamb=${lamb}, lmd_consec=${lmd_consec}"
        echo "----------------------------------------"

        output_dir="save_pt_type_${name}_${variant}"
        log_file="pt_type_${name}_${variant}.log"
        rt_dir="save_rt_type_${variant}"

        # Skip if already trained (resume support)
        if [ "$1" == "-r" ] && [ -f "${output_dir}/test_result.txt" ]; then
            echo "  SKIPPED (already complete)"
            continue
        fi

        # Verify RT checkpoint exists
        latest_rt=$(ls "$rt_dir/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
        if [ -z "$latest_rt" ]; then
            echo "  ERROR: No RT checkpoint found in ${rt_dir}, skipping"
            continue
        fi
        rt_epoch=$(basename "$latest_rt" | grep -oE '[0-9]+')
        echo "  Using RT checkpoint: duorec-${rt_epoch}.pth"

        # Get latest PT checkpoint (for resume)
        start_epoch=1
        if [ -d "${output_dir}/model" ]; then
            latest_pt=$(ls "${output_dir}/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
            if [ -n "$latest_pt" ]; then
                start_epoch=$(($(basename "$latest_pt" | grep -oE '[0-9]+') + 1))
                echo "  Resuming from epoch ${start_epoch}"
            fi
        fi

        # Build args
        div_flag=""
        if [ "$lamb" != "0" ]; then
            div_flag="-div -lamb ${lamb}"
        fi

        consec_flag=""
        if [ "$lmd_consec" != "0" ]; then
            consec_flag="-lmd_consec ${lmd_consec}"
        fi

        # Run training
        python3 main_pt.py \
            -tf ./KuaiRec_variants/${variant}/train-v0.txt \
            -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
            -ef ./KuaiRec_variants/${variant}/test-v0.txt \
            -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
            -n 10728 -n_cat 31 -e 500 -b 64 -l 5e-4 \
            ${div_flag} ${consec_flag} \
            -t_mode topk \
            -early_stop -patience 50 -min_delta 0.0001 \
            -start_epoch ${start_epoch} -epoch_step 1 \
            -i ./${rt_dir} \
            -o ./${output_dir} \
            2>&1 | tee "$log_file"

        echo "  Done: type_${name} / ${variant}"
        echo ""
    done
done

echo ""
echo "============================================"
echo "Type embedding pipeline complete!"
echo "RT results in: save_rt_type_<variant>/"
echo "PT results in: save_pt_type_*_<variant>/test_result.txt"
echo "============================================"
