#!/usr/bin/env python3
"""
Convert MicroLens (sampled) to the Yelp-style format used by the TRIER pipeline.

Input  (datasets/MicroLens/):
  interaction_sampled.csv : user_id,pid,...,category_id,...,exposed_time,watch_time,duration,...,click,...
  categories_cn_en.csv    : category_id -> names (used only for a reference dump)

Preprocessing (per user request: KuaiRec-style engagement filter):
  - watch_ratio = watch_time / duration, keep interactions with ratio >= 0.1
    (mirrors kuairec_highest_individual: individual filter + dedup)
  - dedup user-video pairs (keep highest watch_ratio, tie -> earliest time)
  - re-index item IDs to 1..N (0 reserved for padding)
  - leave-last-2 time split identical to convert_kuairec_to_yelp.py
  - category_id re-indexed to 0..n_cat-1, multi-hot microlens_vec.npy
  - 99 negatives per test sequence (seed 4444)

Outputs (MicroLens/):
  train-v0.txt, valid-v0.txt, test-v0.txt
  microlens_cate.txt, microlens_vec.npy
  MicroLens-random-sample_size=99-seed=4444.txt
"""
import os
import csv
import argparse
import numpy as np
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description='Convert MicroLens to Yelp format')
    ap.add_argument('--input_dir', default='./datasets/MicroLens')
    ap.add_argument('--output_dir', default='./MicroLens')
    ap.add_argument('--min_watch_ratio', type=float, default=0.1,
                    help='KuaiRec-style engagement threshold (default 0.1)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_csv = os.path.join(args.input_dir, 'interaction_sampled.csv')

    # ---------- read interactions, compute watch_ratio ----------
    interactions = []
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                watch_time = float(row['watch_time'])
                duration = float(row['duration'])
                ratio = watch_time / duration if duration > 0 else 0.0
            except (ValueError, KeyError):
                ratio = 0.0
            if ratio < args.min_watch_ratio:
                continue
            # lexicographic sort of "YYYY-MM-DD HH:MM:SS" == chronological
            interactions.append({
                'user': int(row['user_id']),
                'item': int(row['pid']),
                'time': str(row['exposed_time']),
                'ratio': ratio,
            })
    print(f"Interactions after watch_ratio >= {args.min_watch_ratio}: {len(interactions)}")

    # ---------- dedup user-video pairs (highest watch_ratio) ----------
    best = {}
    for it in interactions:
        key = (it['user'], it['item'])
        cur = best.get(key)
        if cur is None or (-it['ratio'], it['time']) < (-cur['ratio'], cur['time']):
            best[key] = it
    dedup = list(best.values())
    print(f"After dedup: {len(dedup)}")

    # ---------- sequences ----------
    user_seqs = defaultdict(list)
    for it in dedup:
        user_seqs[it['user']].append((it['time'], it['item']))
    all_lines = []
    for u in sorted(user_seqs):
        seq = [i for _, i in sorted(user_seqs[u])]
        if len(seq) >= 2:
            all_lines.append((u, seq))
    print(f"Users with >= 2 items: {len(all_lines)}")

    # ---------- re-index items ----------
    used_items = sorted({i for _, seq in all_lines for i in seq})
    item_map = {raw: new for new, raw in enumerate(used_items, start=1)}
    n_items = len(used_items) + 1
    print(f"Re-indexed items: {len(used_items)}  -> item_num (-n) = {n_items}")

    # ---------- split ----------
    train_lines, valid_lines, test_lines = [], [], []
    for u, seq in all_lines:
        s = [str(item_map[i]) for i in seq]
        if len(s) >= 3:
            train_lines.append(f"{u} {' '.join(s[:-2])}")
            valid_lines.append(f"{u} {' '.join(s[:-1])}")
            test_lines.append(f"{u} {' '.join(s)}")
        else:
            train_lines.append(f"{u} {s[0]}")
            valid_lines.append(f"{u} {' '.join(s)}")
            test_lines.append(f"{u} {' '.join(s)}")

    for name, lines in [('train-v0.txt', train_lines), ('valid-v0.txt', valid_lines),
                        ('test-v0.txt', test_lines)]:
        with open(os.path.join(args.output_dir, name), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Wrote {len(lines):6d} lines to {name}")

    # ---------- categories (leaf category_id from interaction rows) ----------
    raw_item_cat = {}
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                raw_item_cat.setdefault(int(row['pid']), int(row['category_id']))
            except (ValueError, KeyError):
                pass
    cat_names = sorted(set(raw_item_cat.values()))
    cat_map = {c: idx for idx, c in enumerate(cat_names)}
    n_cat = len(cat_names)
    print(f"Unique categories: {n_cat}  -> -n_cat {n_cat}")

    cate_lines = []
    for raw, new in sorted(item_map.items(), key=lambda kv: kv[1]):
        c = raw_item_cat.get(raw)
        if c is None:
            continue
        cate_lines.append(f"{new} {cat_map[c]}")
    with open(os.path.join(args.output_dir, 'microlens_cate.txt'), 'w') as f:
        f.write('\n'.join(cate_lines) + '\n')

    # ---------- multi-hot vec.npy (single-category -> one-hot) ----------
    item_vecs = np.zeros((n_items, n_cat), dtype=np.float32)
    for raw, new in item_map.items():
        c = raw_item_cat.get(raw)
        if c is not None:
            item_vecs[new, cat_map[c]] = 1.0
    np.save(os.path.join(args.output_dir, 'microlens_vec.npy'), item_vecs)
    print(f"Saved microlens_vec.npy with shape {item_vecs.shape}")

    # ---------- negatives ----------
    generate_negatives(os.path.join(args.output_dir, 'test-v0.txt'),
                       os.path.join(args.output_dir, 'MicroLens-random-sample_size=99-seed=4444.txt'),
                       n_items)

    print("\nConversion complete! Run with: -n {} -n_cat {}".format(n_items, n_cat))


def generate_negatives(test_file, output_neg_file, item_num):
    sequences = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                sequences.append(set(map(int, line.split()[1:])))
    rng = np.random.default_rng(4444)
    all_items = np.arange(1, item_num)
    negatives = []
    for seq_set in sequences:
        candidates = rng.choice(all_items, size=250, replace=False)
        neg = [int(c) for c in candidates if int(c) not in seq_set][:99]
        while len(neg) < 99:
            extra = rng.choice(all_items, size=99, replace=False)
            neg.extend([int(c) for c in extra if int(c) not in seq_set])
            neg = neg[:99]
        negatives.append(' '.join(map(str, neg)))
    with open(output_neg_file, 'w') as f:
        f.write('\n'.join(negatives) + '\n')
    print(f"Wrote 99-negatives for {len(negatives)} test sequences")


if __name__ == '__main__':
    main()
