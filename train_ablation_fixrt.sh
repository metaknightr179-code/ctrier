#!/bin/bash
# =============================================================================
# Ablation: isolate which "fix" hurt the lambda>0 configs.
# Reference point: old broken-RT type lamb0005 (no mask, b64, lr 5e-4)
#   greedy protocol mean R@20 = 0.1015 (best 0.1154)
# vs new fixrt type lamb0005 (mask, b256, lr 1e-3): 0.0723 (n=2)
#
# Trains PT (type lamb0005) with the FIXED RT but varying mask/hyperparams:
#   B: mask   + b64/lr5e-4   -> isolates hyperparams (vs A)
#   C: noMask + b256/lr1e-3  -> isolates mask (vs A)
#   D: noMask + b64/lr5e-4   -> old-style setup with fixed RT
#   A: mask   + b256/lr1e-3  = already trained (save_pt_fixrt_lamb0005_*)
#
# Uses existing fixed RT checkpoints (save_rt_fix_<variant>).
# Usage: nohup bash train_ablation_fixrt.sh > train_ablation.log 2>&1 &
# =============================================================================
VARIANTS=(
    "kuairec_highest_individual"
    "kuairec_first_individual"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

GPU=0
MAX_EPOCHS=500

run_pt() {
    local TAG=$1 VAR=$2 BS=$3 LR=$4 EXTRA=$5
    local PT_DIR="./save_pt_abl${TAG}_${VAR}"
    local RT_DIR="./save_rt_fix_${VAR}"

    echo "=============================================="
    echo "PT abl${TAG}: ${VAR} (b=${BS}, lr=${LR}, ${EXTRA:-baseline})"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
        -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
        -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
        -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
        -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m train -e ${MAX_EPOCHS} -b ${BS} -l ${LR} \
        -div -lamb 0.005 -lmd_consec 0 -t_mode topk \
        -i ${RT_DIR} -o ${PT_DIR} ${EXTRA} 2>&1 | tee "train_pt_abl${TAG}_${VAR}.log"
}

for VAR in "${VARIANTS[@]}"; do
    run_pt B "${VAR}" 64  5e-4 ""
    run_pt C "${VAR}" 256 1e-3 "-no_mask"
    run_pt D "${VAR}" 64  5e-4 "-no_mask"
done

# Greedy eval of all ablation checkpoints
for VAR in "${VARIANTS[@]}"; do
    for TAG in B C D; do
        PT_DIR="./save_pt_abl${TAG}_${VAR}"
        RT_DIR="./save_rt_fix_${VAR}"
        LATEST=$(ls "${PT_DIR}"/model/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1)
        [ -z "$LATEST" ] && continue
        echo "=== greedy eval abl${TAG}/${VAR} (epoch ${LATEST}) ==="
        CUDA_VISIBLE_DEVICES=${GPU} python3 main_pt.py \
            -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
            -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
            -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
            -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
            -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
            -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
            -m test -e ${LATEST} -b 256 -lamb 0.005 \
            -div -lmd_consec 0 -t_mode greedy \
            -start_epoch ${LATEST} -epoch_step 1 \
            -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_greedy_abl${TAG}_${VAR}.log"
    done
done

echo "ABLATION COMPLETE — results in save_pt_abl{B,C,D}_*/test_result.txt"
