#!/usr/bin/env python3
"""
Compute ILD@20 and CS@20 for SASRec on the small-matrix test splits.

sasrec_pytorch.py's evaluate only reports recall/MRR/NDCG, so this script
regenerates each user's top-20 list exactly as evaluate_sasrec does (same
input masking + ranking, via recalc_cs_baselines.gen_sasrec_lists) and scores
it with the same ILD/CS formulas that script.py applies to TRIER-PT/GRU4Rec
(item vectors = KuaiRec_variants/kuairec_vec.npy).

Run AFTER eval_small_baselines.sh:
  python3 eval_sasrec_ild_cs_small.py
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from recalc_cs_baselines import gen_sasrec_lists, load_test_lines
from sasrec_pytorch import SASRecModel
from script import cal_ILD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--item_num', type=int, default=10728)
    ap.add_argument('--maxlen', type=int, default=50)
    ap.add_argument('--vec', default='./KuaiRec_variants/kuairec_vec.npy')
    ap.add_argument('--ckpt', default='./sasrec_best.pth')
    ap.add_argument('--variants', nargs='+', default=[
        'kuairec_highest_individual', 'kuairec_highest_average',
        'kuairec_first_individual', 'kuairec_first_average'])
    ap.add_argument('--output', default='sasrec_small_ild_cs.txt')
    args = ap.parse_args()

    device = 'cpu'
    model = SASRecModel(args.item_num, hidden_units=50, num_blocks=2,
                        num_heads=1, dropout_rate=0.5, maxlen=args.maxlen).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    vecs = torch.tensor(np.load(args.vec), dtype=torch.float32)
    if vecs.shape[0] < args.item_num:
        pad = torch.zeros(args.item_num - vecs.shape[0], vecs.shape[1])
        vecs = torch.cat([vecs, pad], dim=0)

    rows = []
    for var in args.variants:
        test_file = f'./KuaiRec_small_eval/{var}/test-v0.txt'
        seqs = load_test_lines(test_file)
        rec_lists = gen_sasrec_lists(model, seqs, args.item_num, args.maxlen, device)

        ilds, css = [], []
        for rec in rec_lists:
            ids = torch.tensor(rec[:20], dtype=torch.long)
            ilds.append(cal_ILD(vecs[ids], 20))
            v = vecs[ids]
            css.append(F.cosine_similarity(v[:-1], v[1:], dim=-1).mean().item())
        row = (var, len(rec_lists), float(np.mean(ilds)), float(np.mean(css)))
        rows.append(row)
        print(f'{var}: n={row[1]} ILD@20={row[2]:.4f} CS@20={row[3]:.4f}')

    with open(args.output, 'w') as f:
        f.write('variant\tn_users\tILD@20\tCS@20\n')
        for var, n, ild, cs in rows:
            f.write(f'{var}\t{n}\t{ild:.6f}\t{cs:.6f}\n')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
