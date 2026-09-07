#!/usr/bin/env python3
"""
Convert KuaiRand-1K to the Yelp-style format used by the TRIER pipeline.

Input  (datasets/KuaiRand-1k/):
  standard_interactions.csv : user_id,video_id,time_ms,is_click,...,long_view
  item_features_mapped.csv  : video_id,music_id,author_id,video_duration,tag,... (tag = comma-separated category ids)

Preprocessing (per user request):
  - keep only interactions with long_view == 1
  - dedup user-video pairs (keep earliest timestamp occurrence)
  - iterative 5-core filtering (users AND items with >= 5 interactions)
    (mandatory here: raw catalog is 4.37M items against 1K users)
  - re-index item IDs to 1..N (0 reserved for padding)
  - leave-last-2 time split identical to convert_kuairec_to_yelp.py
  - tag ids re-indexed to 0..n_cat-1, multi-hot kuairand_vec.npy
  - 99 negatives per test sequence (seed 4444)

Outputs (KuaiRand1K/):
  train-v0.txt, valid-v0.txt, test-v0.txt
  kuairand_cate.txt, kuairand_vec.npy
  KuaiRand-random-sample_size=99-seed=4444.txt
"""
import os
import csv
import argparse
import numpy as np
from collections import defaultdict


def five_core(interactions, min_count=5):
    while True:
        u_cnt = defaultdict(int)
        i_cnt = defaultdict(int)
        for u, i, _ in interactions:
            u_cnt[u] += 1
            i_cnt[i] += 1
        keep_u = {u for u, c in u_cnt.items() if c >= min_count}
        keep_i = {i for i, c in i_cnt.items() if c >= min_count}
        filtered = [(u, i, ts) for u, i, ts in interactions if u in keep_u and i in keep_i]
        if len(filtered) == len(interactions):
            return filtered
        interactions = filtered


def main():
    ap = argparse.ArgumentParser(description='Convert KuaiRand-1K to Yelp format')
    ap.add_argument('--input_dir', default='./datasets/KuaiRand-1k')
    ap.add_argument('--output_dir', default='./KuaiRand1K')
    ap.add_argument('--min_core', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- interactions: long_view == 1 ----------
    raw_inter = []
    with open(os.path.join(args.input_dir, 'standard_interactions.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['long_view'] == '1':
                raw_inter.append((int(row['user_id']), int(row['video_id']),
                                  int(float(row['time_ms']))))
    print(f"long_view interactions: {len(raw_inter)}")

    # ---------- dedup user-video pairs (earliest timestamp) ----------
    best = {}
    for u, i, ts in raw_inter:
        key = (u, i)
        if key not in best or ts < best[key]:
            best[key] = ts
    dedup = [(u, i, ts) for (u, i), ts in best.items()]
    print(f"After dedup: {len(dedup)}")

    # ---------- 5-core ----------
    dedup = five_core(dedup, args.min_core)
    print(f"After {args.min_core}-core: {len(dedup)} interactions")

    # ---------- re-index items ----------
    items = sorted({i for _, i, _ in dedup})
    item_map = {raw: new for new, raw in enumerate(items, start=1)}
    n_items = len(items) + 1
    print(f"Re-indexed items: {len(items)}  -> item_num (-n) = {n_items}")

    # ---------- sequences ----------
    user_seqs = defaultdict(list)
    for u, i, ts in dedup:
        user_seqs[u].append((ts, item_map[i]))
    all_lines = []
    for u in sorted(user_seqs):
        seq = [i for _, i in sorted(user_seqs[u])]
        if len(seq) >= 2:
            all_lines.append((u, seq))
    print(f"Users with >= 2 items: {len(all_lines)}")

    # ---------- split ----------
    train_lines, valid_lines, test_lines = [], [], []
    for u, seq in all_lines:
        s = [str(x) for x in seq]
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

    # ---------- categories (tags) ----------
    tags_by_raw = {}
    with open(os.path.join(args.input_dir, 'item_features_mapped.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = int(row['video_id'])
            tag_str = (row.get('tag') or '').strip()
            if tag_str:
                try:
                    tags_by_raw[raw_id] = [int(t) for t in tag_str.split(',')]
                except ValueError:
                    pass
    used_raw = {raw for raw, new in item_map.items() if raw in tags_by_raw}
    tag_names = sorted({t for raw in used_raw for t in tags_by_raw[raw]})
    tag_map = {t: idx for idx, t in enumerate(tag_names)}
    n_cat = len(tag_names)
    print(f"Unique tags: {n_cat}  -> -n_cat {n_cat}")
    n_untagged = sum(1 for raw in item_map if raw not in tags_by_raw)
    if n_untagged:
        print(f"Warning: {n_untagged} items have no tags (get empty category set)")

    cate_lines = []
    for raw, new in sorted(item_map.items(), key=lambda kv: kv[1]):
        cats = sorted(tag_map[t] for t in tags_by_raw.get(raw, []))
        cate_lines.append(f"{new} {' '.join(map(str, cats))}")
    with open(os.path.join(args.output_dir, 'kuairand_cate.txt'), 'w') as f:
        f.write('\n'.join(cate_lines) + '\n')

    # ---------- multi-hot vec.npy ----------
    item_vecs = np.zeros((n_items, n_cat), dtype=np.float32)
    for raw, new in item_map.items():
        for t in tags_by_raw.get(raw, []):
            item_vecs[new, tag_map[t]] = 1.0
    np.save(os.path.join(args.output_dir, 'kuairand_vec.npy'), item_vecs)
    print(f"Saved kuairand_vec.npy with shape {item_vecs.shape}")

    # ---------- negatives ----------
    generate_negatives(os.path.join(args.output_dir, 'test-v0.txt'),
                       os.path.join(args.output_dir, 'KuaiRand-random-sample_size=99-seed=4444.txt'),
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
