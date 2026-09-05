#!/bin/bash
# =============================================================================
# GREEDY-mode (lambda-aware) evaluation — the original TRIER inference protocol.
# Uses the full generate_by_score path (RT augmentation + lamb blending),
# NOT topk which bypasses it.
#
# Evaluates:
#   - new fixed-RT checkpoints: save_pt_fixrt_<cfg>_<var>   (RT: save_rt_fix_<var>)
#   - old broken-RT checkpoints: save_pt_type_lamb0005_<var> (RT: save_rt_type_<var>)
# =============================================================================
cd /Users/notrobin/Documents/trae_projects/trier

# PT_DIR|RT_DIR|LAMB|VARIANT
TARGETS=(
    "save_pt_fixrt_nodiv_kuairec_highest_individual|save_rt_fix_kuairec_highest_individual|0|kuairec_highest_individual"
    "save_pt_fixrt_nodiv_kuairec_highest_average|save_rt_fix_kuairec_highest_average|0|kuairec_highest_average"
    "save_pt_fixrt_nodiv_kuairec_first_individual|save_rt_fix_kuairec_first_individual|0|kuairec_first_individual"
    "save_pt_fixrt_nodiv_kuairec_first_average|save_rt_fix_kuairec_first_average|0|kuairec_first_average"
    "save_pt_fixrt_lamb0002_kuairec_highest_individual|save_rt_fix_kuairec_highest_individual|0.002|kuairec_highest_individual"
    "save_pt_fixrt_lamb0002_kuairec_highest_average|save_rt_fix_kuairec_highest_average|0.002|kuairec_highest_average"
    "save_pt_fixrt_lamb0002_kuairec_first_individual|save_rt_fix_kuairec_first_individual|0.002|kuairec_first_individual"
    "save_pt_fixrt_lamb0002_kuairec_first_average|save_rt_fix_kuairec_first_average|0.002|kuairec_first_average"
    "save_pt_fixrt_lamb0005_kuairec_highest_individual|save_rt_fix_kuairec_highest_individual|0.005|kuairec_highest_individual"
    "save_pt_fixrt_lamb0005_kuairec_highest_average|save_rt_fix_kuairec_highest_average|0.005|kuairec_highest_average"
    "save_pt_type_lamb0005_kuairec_highest_individual|save_rt_type_kuairec_highest_individual|0.005|kuairec_highest_individual"
    "save_pt_type_lamb0005_kuairec_highest_average|save_rt_type_kuairec_highest_average|0.005|kuairec_highest_average"
    "save_pt_type_lamb0005_kuairec_first_individual|save_rt_type_kuairec_first_individual|0.005|kuairec_first_individual"
    "save_pt_type_lamb0005_kuairec_first_average|save_rt_type_kuairec_first_average|0.005|kuairec_first_average"
)

get_latest_epoch() {
    ls "${1}"/duorec-*.pth 2>/dev/null | sed 's/.*duorec-//;s/\.pth//' | sort -n | tail -1
}

for T in "${TARGETS[@]}"; do
    IFS='|' read -r PT_DIR RT_DIR LAMB VAR <<< "$T"

    if [ ! -d "$PT_DIR" ]; then echo "SKIP: missing $PT_DIR"; continue; fi

    LATEST=$(get_latest_epoch "${PT_DIR}/model")
    if [ -z "$LATEST" ]; then echo "SKIP: no checkpoint in $PT_DIR"; continue; fi

    # div flag: div_loss was active in training for all lambda>0 configs
    if [ "$LAMB" == "0" ]; then DIV_FLAG=""; else DIV_FLAG="-div -lamb ${LAMB}"; fi

    TAG=$(echo "$PT_DIR" | sed 's|save_pt_||')

    echo "=============================================="
    echo "GREEDY eval: ${TAG} (epoch ${LATEST}, lamb=${LAMB})"
    echo "=============================================="

    # keep topk results (main_pt.py writes test_result.txt); back it up first
    [ -f "${PT_DIR}/test_result.txt" ] && mv "${PT_DIR}/test_result.txt" "${PT_DIR}/test_result_topk.txt"
    python3 main_pt.py \
        -tf ./KuaiRec_variants/${VAR}/train-v0.txt \
        -vf ./KuaiRec_variants/${VAR}/valid-v0.txt \
        -ef ./KuaiRec_variants/${VAR}/test-v0.txt \
        -vn ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -en ./KuaiRec_variants/${VAR}/KuaiRec-random-sample_size=99-seed=4444.txt \
        -cat ./KuaiRec_variants/${VAR}/kuairec_cate.txt \
        -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
        -m test -e ${LATEST} -b 256 \
        ${DIV_FLAG} -lmd_consec 0 -t_mode greedy \
        -start_epoch ${LATEST} -epoch_step 1 \
        -i ${RT_DIR} -o ${PT_DIR} 2>&1 | tee "eval_greedy_${TAG}.log"

    echo "=== ${TAG} done ==="
    echo ""
done

echo "ALL GREEDY EVALS DONE"
