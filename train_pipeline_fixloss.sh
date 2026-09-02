#!/bin/bash
# =============================================================================
# TRIER Training Pipeline: ALL λ values (FIXED diversity loss)
#
# Trains all configs with the ORIGINAL GitHub diversity loss formula:
#   div_loss = -(ILD_greedy - ILD_ture) * gate
#
# Output dirs use "_fixloss" suffix to avoid overwriting old checkpoints.
#
# Type-embedding models:   save_pt_type_fixloss_<config>_<variant>
# No-type models:          save_pt_fixloss_<config>_<variant>
#
# RT checkpoints (REUSE EXISTING — RT loss unchanged):
#   Type:    save_rt_type_<variant>
#   No-type: save_rt_kuairec_<variant#kuairec_>
#
# Usage (run in the project directory where all files live):
#   bash train_pipeline_fixloss.sh
#   nohup bash train_pipeline_fixloss.sh > train_fixloss.log 2>&1 &
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
CONFIGS=(
    "nodiv|0|0"
    "lamb0002|0.002|0"
    "lamb0005|0.005|0"
    "lamb001|0.01|0"
    "lamb005|0.05|0"
    "lamb01|0.1|0"
)

OUT_PREFIX="fixloss"

echo "############################################################"
echo "# TRIER PT Training — FIXED diversity loss (original formula)"
echo "# Output prefix: save_pt[_type]_fixloss_*"
echo "# $((${#CONFIGS[@]} * ${#VARIANTS[@]} * 2)) total runs"
echo "############################################################"
echo ""

for config_line in "${CONFIGS[@]}"; do
    IFS='|' read -r name lamb lmd_consec <<< "$config_line"

    for variant in "${VARIANTS[@]}"; do
        echo "============================================================"
        echo "Training: ${name} / ${variant} (lamb=${lamb})"
        echo "============================================================"

        # ---- TYPE-EMBEDDING VERSION ----
        type_dir="save_pt_type_${OUT_PREFIX}_${name}_${variant}"
        type_log="pt_type_${OUT_PREFIX}_${name}_${variant}.log"
        rt_type_dir="save_rt_type_${variant}"

        echo "  [TYPE] -> ${type_dir}"
        start_epoch=1
        if [ -d "${type_dir}/model" ]; then
            latest=$(ls "${type_dir}/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
            if [ -n "$latest" ]; then
                start_epoch=$(($(basename "$latest" | grep -oE '[0-9]+') + 1))
                echo "    Resuming from epoch ${start_epoch}"
            fi
        fi

        if [ "${lamb}" = "0" ]; then
            python3 main_pt.py \
                -tf ./KuaiRec_variants/${variant}/train-v0.txt \
                -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
                -ef ./KuaiRec_variants/${variant}/test-v0.txt \
                -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
                -n 10728 -n_cat 31 -e 500 -b 64 -l 5e-4 \
                -t_mode topk \
                -early_stop -patience 50 -min_delta 0.0001 \
                -start_epoch ${start_epoch} -epoch_step 1 \
                -i ./${rt_type_dir} \
                -o ./${type_dir} \
                2>&1 | tee "$type_log"
        else
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
        fi

        echo "  [TYPE] Done: ${name} / ${variant}"
        echo ""

        # ---- NO-TYPE VERSION ----
        notype_dir="save_pt_${OUT_PREFIX}_${name}_${variant}"
        notype_log="pt_${OUT_PREFIX}_${name}_${variant}.log"
        rt_notype_dir="save_rt_kuairec_${variant#kuairec_}"

        echo "  [NO-TYPE] -> ${notype_dir}"
        start_epoch=1
        if [ -d "${notype_dir}/model" ]; then
            latest=$(ls "${notype_dir}/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
            if [ -n "$latest" ]; then
                start_epoch=$(($(basename "$latest" | grep -oE '[0-9]+') + 1))
                echo "    Resuming from epoch ${start_epoch}"
            fi
        fi

        if [ "${lamb}" = "0" ]; then
            python3 main_pt.py \
                -tf ./KuaiRec_variants/${variant}/train-v0.txt \
                -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
                -ef ./KuaiRec_variants/${variant}/test-v0.txt \
                -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
                -n 10728 -n_cat 31 -e 500 -b 64 -l 5e-4 \
                -t_mode topk \
                -early_stop -patience 50 -min_delta 0.0001 \
                -start_epoch ${start_epoch} -epoch_step 1 \
                -i ./${rt_notype_dir} \
                -o ./${notype_dir} \
                2>&1 | tee "$notype_log"
        else
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
        fi

        echo "  [NO-TYPE] Done: ${name} / ${variant}"
        echo ""
    done
done

echo ""
echo "############################################################"
echo "Training complete!"
echo "############################################################"
echo ""
echo "Type-embedding results:    save_pt_type_fixloss_*/test_result.txt"
echo "No-type results:           save_pt_fixloss_*/test_result.txt"
echo ""
echo "Next: evaluate with:"
echo "  for d in save_pt_fixloss_*/model save_pt_type_fixloss_*/model; do"
echo "    EPOCH=\$(ls \$d/duorec-*.pth | sed 's/.*duorec-//;s/.pth//' | sort -n | tail -1)"
echo "    echo \"\$d -> epoch \$EPOCH\""
echo "  done"
