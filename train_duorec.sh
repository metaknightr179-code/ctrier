#!/bin/bash
# =============================================================================
# DuoRec backbone training (RT-free, diversity-free)
#
# DuoRec = the TRIER-PT backbone without TRIER additions:
#   - NO -div      -> diversity loss off; RT model is NEVER called in training
#   - NO type embeddings (-no_type) -> plain item ID embeddings (faithful DuoRec)
#   - SSL contrastive loss ON (default -ssl us_x: sequences sharing the same
#     target item are positive pairs) — this IS DuoRec's core mechanism
#   - Causal mask ON (SASRec-style autoregressive encoder)
#   - Inference: -t_mode topk (standard full-ranking next-item; no RT greedy)
#
# Hyperparameters match original TRIER/DuoRec defaults: -b 256 -l 1e-3
# RT checkpoint path (-i) is still passed (main_pt.py always constructs the RT
# model), but it is never invoked without -div / under topk eval.
#
# Usage (server, GPU):
#   nohup bash train_duorec.sh > train_duorec.log 2>&1 &
#   tail -f train_duorec.log
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

GPU=0
MAX_EPOCHS=500

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

echo "############################################################"
echo "# DuoRec backbone training (no RT, no div loss, no types)"
echo "# Variants: ${#VARIANTS[@]} | epochs: ${MAX_EPOCHS} | -b 256 -l 1e-3"
echo "############################################################"
echo ""

for variant in "${VARIANTS[@]}"; do
    out_dir="save_duorec_${variant}"
    log_file="duorec_${variant}.log"
    rt_dir="save_rt_fix_${variant}"

    echo "=============================================="
    echo "DuoRec Training -> ${out_dir}"
    echo "=============================================="

    RESUME=""
    if [ -f "${out_dir}/train_result.txt" ]; then
        EPOCHS_DONE=$(wc -l < "${out_dir}/train_result.txt")
        if [ "$EPOCHS_DONE" -ge "$MAX_EPOCHS" ]; then
            echo "  Already complete (${EPOCHS_DONE} epochs) - skip"
            echo ""
            continue
        fi
        if [ -f "${out_dir}/model/duorec-$((EPOCHS_DONE - 1)).pth" ]; then
            echo "  Resuming from epoch ${EPOCHS_DONE}"
            RESUME="-r"
        else
            echo "  Stale train_result.txt (no checkpoint) - starting fresh"
            rm -f "${out_dir}/train_result.txt"
        fi
    fi

    # -i points at an isolated dummy dir so we NEVER accidentally load a
    # partially-trained RT checkpoint from a parallel TRIER pipeline run.
    # main_pt.py finds no checkpoint there -> warns -> random RT (never invoked:
    # no -div + topk eval), which is exactly what DuoRec wants.
    rt_dir="rt_dummy_for_duorec"
    mkdir -p "${rt_dir}"

    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf ./KuaiRec_variants/${variant}/train-v0.txt \
        -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
        -ef ./KuaiRec_variants/${variant}/test-v0.txt \
        -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m train -e ${MAX_EPOCHS} -b 256 -l 1e-3 \
        -no_type \
        -t_mode topk \
        -early_stop -patience 50 -min_delta 0.0001 \
        ${RESUME} \
        -i ./${rt_dir} \
        -o ./${out_dir} 2>&1 | tee "${log_file}"

    echo "  DuoRec done: ${out_dir}"
    echo ""
done

echo "############################################################"
echo "DUOREC TRAINING COMPLETE"
echo "Checkpoints: save_duorec_<variant>/"
echo "Next: bash eval_duorec.sh"
echo "############################################################"
