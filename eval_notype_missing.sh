#!/bin/bash
cd "$(dirname "$0")"

# Only highest_individual has RT checkpoints locally (save_rt_highest_individual)
# The other 3 variants need RT checkpoints transferred from remote

echo "=== Eval: lamb005/kuairec_highest_individual (epoch 500) ==="
python3 main_pt.py \
  -tf ./KuaiRec_variants/kuairec_highest_individual/train-v0.txt \
  -vf ./KuaiRec_variants/kuairec_highest_individual/valid-v0.txt \
  -ef ./KuaiRec_variants/kuairec_highest_individual/test-v0.txt \
  -vn ./KuaiRec_variants/kuairec_highest_individual/KuaiRec-random-sample_size=99-seed=4444.txt \
  -en ./KuaiRec_variants/kuairec_highest_individual/KuaiRec-random-sample_size=99-seed=4444.txt \
  -cat ./KuaiRec_variants/kuairec_highest_individual/kuairec_cate.txt \
  -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
  -m test -e 500 -b 64 \
  -div -lamb 0.05 -lmd_consec 0 -t_mode topk \
  -start_epoch 500 -epoch_step 1 \
  -i ./save_rt_highest_individual -o ./save_pt_lamb005_kuairec_highest_individual \
  2>&1 | tail -5
echo "--- Done: lamb005/kuairec_highest_individual ---"

echo "=== Eval: lamb01/kuairec_highest_individual (epoch 19, barely trained) ==="
python3 main_pt.py \
  -tf ./KuaiRec_variants/kuairec_highest_individual/train-v0.txt \
  -vf ./KuaiRec_variants/kuairec_highest_individual/valid-v0.txt \
  -ef ./KuaiRec_variants/kuairec_highest_individual/test-v0.txt \
  -vn ./KuaiRec_variants/kuairec_highest_individual/KuaiRec-random-sample_size=99-seed=4444.txt \
  -en ./KuaiRec_variants/kuairec_highest_individual/KuaiRec-random-sample_size=99-seed=4444.txt \
  -cat ./KuaiRec_variants/kuairec_highest_individual/kuairec_cate.txt \
  -n 10728 -n_cat 31 -vec ./KuaiRec_variants/kuairec_vec.npy \
  -m test -e 19 -b 64 \
  -div -lamb 0.1 -lmd_consec 0 -t_mode topk \
  -start_epoch 19 -epoch_step 1 \
  -i ./save_rt_highest_individual -o ./save_pt_lamb01_kuairec_highest_individual \
  2>&1 | tail -5
echo "--- Done: lamb01/kuairec_highest_individual ---"

echo "=== ALL LOCAL EVALS COMPLETE ==="
echo "NOTE: lamb005 for highest_average, first_individual, first_average"
echo "      need RT checkpoints transferred from remote (save_rt_kuairec_<variant>)"
