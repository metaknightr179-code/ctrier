#!/bin/bash
# =============================================================================
# Stage 2 of train_pipeline_fixrt.sh extracted for PARALLEL GPU execution:
# PT models WITH type embeddings (diversity sweep).
#
# Usage:
#   nohup bash train_pt_type_fixrt.sh <GPU_ID> > pt_type_1000.log 2>&1 &
#   e.g.: nohup bash train_pt_type_fixrt.sh 0 > pt_type_1000.log 2>&1 &
#
# Waits for Stage 1 (RT) to finish if main_rt.py is still running, then
# trains 7 configs x 4 variants sequentially on the given GPU. Launch
# train_pt_notype_fixrt.sh with a different GPU ID at the same time to
# run both PT families in parallel.
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

# SUFFIX|LAMB|LMD_CONSEC
CONFIGS=(
    "nodiv|0|0"
    "lamb0002|0.002|0"
    "lamb0005|0.005|0"
    "lamb0005_consec0001|0.005|0.001"
    "lamb001|0.01|0"
    "lamb005|0.05|0"
    "lamb01|0.1|0"
)

echo "############################################################"
echo "# PT [TYPE] stage (with type embeddings), GPU ${GPU}"
echo "# Runs: $((${#CONFIGS[@]} * ${#VARIANTS[@]}))"
echo "# Hyperparams: -b 256 -l 1e-3 -e ${MAX_EPOCHS} -early_stop patience=100"
echo "############################################################"
echo ""

# Wait for the RT stage to FULLY finish: each variant needs either a DONE
# marker (written by main_rt.py on normal completion) or a complete
# train_result.txt (>= MAX_EPOCHS lines, covers RT processes started before
# the marker existed). Never start PT on a partially-trained RT.
# NOTE: if an RT variant early-stopped under old code (no marker, < MAX_EPOCHS
# lines), touch save_rt_fix_<variant>/DONE manually to release this wait.
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

for config_line in "${CONFIGS[@]}"; do
    IFS='|' read -r name lamb lmd_consec <<< "$config_line"

    for variant in "${VARIANTS[@]}"; do
        pt_dir="save_pt_fixrt_${name}_${variant}"
        pt_log="pt_fixrt_${name}_${variant}.log"
        rt_dir="save_rt_fix_${variant}"

        echo "============================================================"
        echo "PT Training [TYPE]: ${name} / ${variant} (lamb=${lamb}) -> ${pt_dir}"
        echo "============================================================"

        RESUME=""
        if [ -f "${pt_dir}/train_result.txt" ]; then
            EPOCHS_DONE=$(wc -l < "${pt_dir}/train_result.txt")
            if [ "$EPOCHS_DONE" -ge "$MAX_EPOCHS" ]; then
                echo "  Already complete (${EPOCHS_DONE} epochs) - skip"
                echo ""
                continue
            fi
            # main_pt.py -r loads duorec-<EPOCHS_DONE>.pth (one log line per epoch)
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
            # -div invokes RT beam generation: require a real RT checkpoint,
            # otherwise a random RT would silently corrupt training
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
            -m train -e ${MAX_EPOCHS} -b 256 -l 1e-3 \
            ${DIV_FLAGS} \
            -t_mode topk \
            -early_stop -patience 100 -min_delta 0.0001 \
            ${RESUME} \
            -i ./${rt_dir} \
            -o ./${pt_dir} 2>&1 | tee "${pt_log}"

        echo "  PT [TYPE] done: ${name} / ${variant}"
        echo ""
    done
done

echo "PT [TYPE] STAGE COMPLETE"
