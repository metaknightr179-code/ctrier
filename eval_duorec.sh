#!/bin/bash
# =============================================================================
# DuoRec evaluation on BOTH protocols:
#   1. big-matrix leave-last-out test (KuaiRec_variants/)   -> test_result.txt
#   2. small-matrix canonical KuaiRec test (KuaiRec_small_eval/) -> test_result_small.txt
#
# Inference: -t_mode topk (DuoRec's natural protocol — plain full-ranking
# next-item; NO RT greedy generation, NO diversity scoring).
# Same arch flags as training: -no_type, no -div.
#
# Usage: bash eval_duorec.sh   (CPU, ~1-2 min per eval; 8 evals total)
# =============================================================================
cd "$(dirname "$0")"

VARIANTS=(
    kuairec_highest_individual
    kuairec_highest_average
    kuairec_first_individual
    kuairec_first_average
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

STAGE_BASE="./save_smalleval_staging"
mkdir -p "$STAGE_BASE"

for variant in "${VARIANTS[@]}"; do
    PT_DIR="./save_duorec_${variant}"
    SMALL_DIR="./KuaiRec_small_eval/${variant}"

    if [ ! -d "$PT_DIR/model" ]; then echo "SKIP: missing $PT_DIR"; continue; fi
    LATEST=$(get_latest_epoch "${PT_DIR}/model")
    if [ -z "$LATEST" ]; then echo "SKIP: no checkpoint in $PT_DIR"; continue; fi
    # DuoRec eval never uses RT (topk, no -div): point -i at an isolated dummy
    # dir so we never auto-load a partially-trained RT checkpoint.
    RT_DIR="./rt_dummy_for_duorec"
    mkdir -p "${RT_DIR}"

    # ------------------------------------------------------------------
    # 1. Big-matrix eval (writes test_result.txt directly in the ckpt dir)
    # ------------------------------------------------------------------
    echo "=============================================="
    echo "BIG-matrix eval: DuoRec / ${variant} (epoch ${LATEST})"
    echo "=============================================="

    python3 main_pt.py \
        -tf ./KuaiRec_variants/${variant}/train-v0.txt \
        -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
        -ef ./KuaiRec_variants/${variant}/test-v0.txt \
        -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        -no_type -t_mode topk \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_duorec_big_${variant}.log"

    # ------------------------------------------------------------------
    # 2. Small-matrix eval (staging dir -> copy result back as test_result_small.txt)
    # ------------------------------------------------------------------
    if [ ! -f "${SMALL_DIR}/test-v0.txt" ]; then
        echo "SKIP small-matrix eval: ${SMALL_DIR}/test-v0.txt missing"
        echo ""
        continue
    fi

    STAGE="${STAGE_BASE}/duorec_${variant}"
    rm -rf "$STAGE"
    mkdir -p "$STAGE"
    ln -s "$(cd "$PT_DIR/model" && pwd)" "$STAGE/model"

    echo "=============================================="
    echo "SMALL-matrix eval: DuoRec / ${variant} (epoch ${LATEST})"
    echo "=============================================="

    python3 main_pt.py \
        -tf ./KuaiRec_variants/${variant}/train-v0.txt \
        -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
        -ef ${SMALL_DIR}/test-v0.txt \
        -vn ${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ${SMALL_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        -no_type -t_mode topk \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i ${RT_DIR} -o ${STAGE} 2>&1 | tee "eval_duorec_small_${variant}.log"

    if [ -f "${STAGE}/test_result.txt" ]; then
        cp "${STAGE}/test_result.txt" "${PT_DIR}/test_result_small.txt"
        echo "=== ${variant} small-matrix done -> ${PT_DIR}/test_result_small.txt ==="
    else
        echo "=== ${variant} small-matrix FAILED (no test_result.txt) ==="
    fi
    echo ""
done

echo "ALL DUOREC EVALS DONE"
