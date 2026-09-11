#!/bin/bash
cd "$(dirname "$0")"

eval_config() {
  config=$1; lamb=$2; lmd_consec=$3; variant=$4; epoch=$5
  pt_dir="save_pt_type_${config}_${variant}"
  rt_dir="save_rt_type_${variant}"

  echo "=== Eval: ${config}/${variant} (epoch ${epoch}) ==="
  python3 main_pt.py \
    -tf ./KuaiRec_variants/${variant}/train-v0.txt \
    -vf ./KuaiRec_variants/${variant}/valid-v0.txt \
    -ef ./KuaiRec_variants/${variant}/test-v0.txt \
    -vn ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
    -en ./KuaiRec_variants/${variant}/KuaiRec-random-sample_size=99-seed=4444.txt \
    -cat ./KuaiRec_variants/${variant}/kuairec_cate.txt \
    -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
    -m test -e ${epoch} -b 64 \
    -div -lamb ${lamb} -lmd_consec ${lmd_consec} -t_mode topk \
    -start_epoch ${epoch} -epoch_step 1 \
    -i ./${rt_dir} -o ./${pt_dir} 2>&1 | tail -5
  echo "--- Done: ${config}/${variant} ---"
}

eval_config lamb001 0.01 0 kuairec_highest_average 500
eval_config lamb001 0.01 0 kuairec_first_individual 500
eval_config lamb001 0.01 0 kuairec_first_average 500
eval_config lamb005 0.05 0 kuairec_highest_individual 500
eval_config lamb005 0.05 0 kuairec_highest_average 500
eval_config lamb005 0.05 0 kuairec_first_individual 500
eval_config lamb005 0.05 0 kuairec_first_average 318
eval_config consec0001 0.01 0.01 kuairec_highest_individual 376

echo "=== ALL EVALS COMPLETE ==="
