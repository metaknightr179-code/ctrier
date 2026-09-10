#!/bin/bash
# =============================================================================
# PT type+duration side-info stage, single setting: lambda = 0.005 (no sweep).
#
# One PT family, trained WITH additive duration-bucket embeddings AND type:
# (-dur_file ./KuaiRec_variants/kuairec_dur.txt -n_dur 8)
# Duration buckets (milliseconds, 1..8): <3s,3-5s,5-7s,7-10s,10-15s,15-30s,
# 30-60s,>60s (0 = unknown padding). See gen_kuairec_dur.py.
#   typedur = type ON + duration ON  -> side info (compare vs type)
#
# Ablation ladder:
#   type (save_pt_dense_lamb0005_*) < typedur (save_pt_typedur_fixrt_lamb0005_*)
#
# 4 runs total: 1 family x 1 config (lamb=0.005) x 4 variants.
#
# RT checkpoints are SHARED with the existing fixrt pipeline (save_rt_fix_<variant>);
# the RT model has no side-info layers. Duration PTs must be trained from scratch
# (new dur_embedding layer is incompatible with old checkpoints).
#
# Usage:
#   nohup bash train_pt_typedur_fixrt.sh <GPU_ID> > pt_typedur_1000.log 2>&1 &
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=${1:-0}
MAX_EPOCHS=1000

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

# SUFFIX|LAMB|LMD_CONSEC  — type+dur uses ONLY lambda = 0.005
CONFIGS=(
    "lamb0005|0.005|0"
)

# FAMILY|DIR_PREFIX|TYPE_FLAG (duration flags appended to every run)
FAMILIES=(
    "typedur|save_pt_typedur_fixrt_|"
)

DUR_FLAGS="-dur_file ./KuaiRec_variants/kuairec_dur.txt -n_dur 8"

echo "############################################################"
echo "# PT [TYPE+DUR] stage (duration-bucket embeddings), GPU ${GPU}"
echo "# Families: ${!FAMILIES[@]}; Runs: $((${#CONFIGS[@]} * ${#VARIANTS[@]} * ${#FAMILIES[@]}))"
echo "# Hyperparams: -b 256 -l 1e-3 -e ${MAX_EPOCHS} -early_stop patience=100"
echo "############################################################"
echo ""

# Wait for the RT stage to FULLY finish (same logic as train_pt_type_fixrt.sh)
for variant in "${VARIANTS[@]}"; do
    marker="save_rt_fix_${variant}/DONE"
    rt_log="save_rt_fix_${variant}/train_result.txt"
    lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
    if [ ! -f "${marker}" ] && [ "${lines}" -lt "${MAX_EPOCHS}" ]; then
        echo "[Wait] RT for ${variant} not finished - waiting (poll every 120s)..."
        while [ ! -f "${marker}" ]; do
            sleep 120
            lines=$(wc -l < "${rt_log}" 2>/dev/null); lines=${lines:-0}
            [ "${lines}" -ge "${MAX_EPOCHS}" ] && break
        done
    fi
    echo "[Wait] RT for ${variant} complete."
done

# Require RT checkpoints for ALL variants up front (-div configs need them)
MISSING=0
for variant in "${VARIANTS[@]}"; do
    if ! ls "save_rt_fix_${variant}/model/"duorec-*.pth >/dev/null 2>&1; then
        echo "ERROR: no RT checkpoint in save_rt_fix_${variant}/model/ - abort."
        MISSING=1
    fi
done
if [ ${MISSING} -eq 1 ]; then
    echo "Train RT first: nohup bash train_rt_fixrt.sh <GPU_ID> > train_rt_1000.log 2>&1 &"
    exit 1
fi

for fam_line in "${FAMILIES[@]}"; do
    IFS='|' read -r fam_name dir_prefix type_flag <<< "$fam_line"

    for config_line in "${CONFIGS[@]}"; do
        IFS='|' read -r name lamb lmd_consec <<< "$config_line"

        for variant in "${VARIANTS[@]}"; do
            pt_dir="${dir_prefix}${name}_${variant}"
            pt_log="pt_${fam_name}_fixrt_${name}_${variant}.log"
            rt_dir="save_rt_fix_${variant}"

            echo "============================================================"
            echo "PT Training [${fam_name}]: ${name} / ${variant} (lamb=${lamb}) -> ${pt_dir}"
            echo "============================================================"

            RESUME=""
            if [ -f "${pt_dir}/train_result.txt" ]; then
                EPOCHS_DONE=$(wc -l < "${pt_dir}/train_result.txt")
                if [ "$EPOCHS_DONE" -ge "$MAX_EPOCHS" ]; then
                    echo "  Already complete (${EPOCHS_DONE} epochs) - skip"
                    echo ""
                    continue
                fi
                if [ -f "${pt_dir}/model/duorec-${EPOCHS_DONE}.pth" ]; then
                    echo "  Resuming from epoch ${EPOCHS_DONE}"
                    RESUME="-r"
                else
                    echo "  ERROR: ${EPOCHS_DONE} epochs logged but model/duorec-${EPOCHS_DONE}.pth missing - abort (progress preserved)."
                    exit 1
                fi
            fi

            DIV_FLAGS=""
            if [ "${lamb}" != "0" ]; then
                latest_rt=$(ls "${rt_dir}/model/duorec-"*.pth 2>/dev/null | sort -t'-' -k2 -n | tail -1)
                if [ -z "${latest_rt}" ]; then
                    echo "  ERROR: no RT checkpoint in ${rt_dir}/model/ - SKIP ${pt_dir}"
                    continue
                fi
                echo "  Using RT checkpoint: $(basename "${latest_rt}")"
                DIV_FLAGS="-div -lamb ${lamb} -lmd_consec ${lmd_consec}"
            fi

            CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
                -tf ./KuaiRec_variants/${variant}/train-v0.txt \
                -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
                -ef ./KuaiRec_variants/${variant}/test-v0.txt \
                -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
                -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
                -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
                ${DUR_FLAGS} ${type_flag} \
                -m train -e ${MAX_EPOCHS} -b 256 -l 1e-3 \
                -dense \
                ${DIV_FLAGS} \
                -t_mode topk \
                -early_stop -patience 100 -min_delta 0.0001 \
                ${RESUME} \
                -i ./${rt_dir} \
                -o ./${pt_dir} 2>&1 | tee "${pt_log}"

            echo "  PT [${fam_name}] done: ${name} / ${variant}"
            echo ""
        done
    done
done

echo "PT [TYPE+DUR] STAGE COMPLETE"
