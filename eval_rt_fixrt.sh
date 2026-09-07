#!/bin/bash
# =============================================================================
# RT evaluation script — evaluates the latest RT checkpoint for each variant.
#
# RT is retrospective: given reversed sequence [sn..s2], predict first item s1.
# Metrics: Recall@10/20, NDCG@10/20, ILD@10, CS@10, Coverage@10.
#
# Usage: nohup bash eval_rt_fixrt.sh [GPU_ID] > eval_rt.log 2>&1 &
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=${1:-0}

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

echo "############################################################"
echo "# RT Evaluation (latest checkpoint per variant), GPU ${GPU}"
echo "############################################################"
echo ""

for VAR in "${VARIANTS[@]}"; do
    rt_dir="save_rt_fix_${VAR}"
    VAR_DIR="./KuaiRec_variants/${VAR}"

    echo "============================================================"
    echo "RT Eval: ${VAR}"
    echo "============================================================"

    # Find latest checkpoint
    if ! ls "${rt_dir}/model/"duorec-*.pth >/dev/null 2>&1; then
        echo "  SKIP: no checkpoints in ${rt_dir}/model/"
        echo ""
        continue
    fi

    LATEST=$(ls "${rt_dir}/model/"duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1)
    echo "  Latest checkpoint: epoch ${LATEST}"

    CUDA_VISIBLE_DEVICES=${GPU} python3 main_rt.py \
        -tf ${VAR_DIR}/train-v0.txt \
        -vf ${VAR_DIR}/valid-v0.txt \
        -ef ${VAR_DIR}/test-v0.txt \
        -vn ${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ${VAR_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ${VAR_DIR}/kuairec_cate.txt \
        -n 10728 -n_cat 31 \
        -m test \
        -e ${LATEST} \
        -start_epoch ${LATEST} \
        -b 256 \
        -o ./${rt_dir} 2>&1 | tee "rt_eval_${VAR}.log"

    echo "  RT Eval done: ${VAR}"
    echo ""
done

echo "RT EVALUATION COMPLETE"
echo ""
echo "Results saved to: save_rt_fix_*/test_result.txt"
echo "Check with: tail -5 save_rt_fix_*/test_result.txt"
