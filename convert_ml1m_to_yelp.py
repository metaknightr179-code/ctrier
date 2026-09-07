#!/usr/bin/env python3
"""
Convert MovieLens ML-1M to the Yelp-style format used by the TRIER pipeline.

Input  (datasets/ml-1m/):
  ratings.dat : UserID::MovieID::Rating::Timestamp
  movies.dat  : MovieID::Title::Genres   (genres '|'-separated)

Preprocessing (per user request):
  - keep only positive ratings (rating >= 4)
  - iterative 5-core filtering (users AND items with >= 5 interactions)
  - re-index item IDs to 1..N (0 reserved for padding)
  - leave-last-2 time split identical to convert_kuairec_to_yelp.py
  - 99 negatives per test sequence (seed 4444)
  - genre indices re-indexed to 0..n_cat-1, multi-hot ml1m_vec.npy

Outputs (ML1M/):
  train-v0.txt, valid-v0.txt, test-v0.txt
  ml1m_cate.txt            ("item_id cat1 cat2 ..." with re-indexed ids)
  ml1m_vec.npy             (multi-hot genre vectors [max_id+1, n_cat])
  ML1M-random-sample_size=99-seed=4444.txt
Prints the recommended -n (item_num) and -n_cat for main_pt.py.
"""
import os
import csv
import argparse
import numpy as np
from collections import defaultdict


def five_core(interactions, min_count=5):
    """Iteratively drop users and items with fewer than min_count interactions.

    Interactions are (user, item, timestamp) triples; timestamps are preserved.
    """
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
    ap = argparse.ArgumentParser(description='Convert ML-1M to Yelp format')
    ap.add_argument('--input_dir', default='./datasets/ml-1m')
    ap.add_argument('--output_dir', default='./ML1M')
    ap.add_argument('--min_rating', type=int, default=4, help='Keep ratings >= min_rating')
    ap.add_argument('--min_core', type=int, default=5, help='5-core threshold')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- ratings ----------
    interactions = []  # (user, item, ts)
    with open(os.path.join(args.input_dir, 'ratings.dat'), 'r') as f:
        for line in f:
            u, i, r, ts = line.strip().split('::')
            r = int(r)
            if r >= args.min_rating:
                interactions.append((int(u), int(i), int(ts)))
    print(f"Positive (>= {args.min_rating}) interactions: {len(interactions)}")

    # ---------- 5-core ----------
    interactions = five_core(interactions, args.min_core)
    print(f"After {args.min_core}-core: {len(interactions)} interactions")

    # ---------- re-index items ----------
    items = sorted({i for _, i, _ in interactions})
    item_map = {raw: new for new, raw in enumerate(items, start=1)}
    n_items = len(items) + 1  # + padding 0
    print(f"Re-indexed items: {len(items)}  -> item_num (-n) = {n_items}")

    # ---------- sequences ----------
    user_seqs = defaultdict(list)
    for u, i, ts in interactions:
        user_seqs[u].append((ts, item_map[i]))
    all_lines = []
    for u in sorted(user_seqs):
        seq = [i for _, i in sorted(user_seqs[u])]
        if len(seq) >= 2:
            all_lines.append((u, seq))
    print(f"Users with >= 2 items: {len(all_lines)}")

    # ---------- split (identical protocol to KuaiRec converter) ----------
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

    # ---------- categories (genres) ----------
    genres_by_raw = {}
    with open(os.path.join(args.input_dir, 'movies.dat'), 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                genres_by_raw[int(parts[0])] = parts[2].split('|')
    used_raw = {raw for raw, new in item_map.items() if raw in genres_by_raw}
    genre_names = sorted({g for raw in used_raw for g in genres_by_raw[raw]})
    genre_map = {g: idx for idx, g in enumerate(genre_names)}
    n_cat = len(genre_names)
    print(f"Genres: {n_cat}  -> -n_cat {n_cat}")

    cate_lines = []
    for raw, new in sorted(item_map.items(), key=lambda kv: kv[1]):
        cats = sorted(genre_map[g] for g in genres_by_raw.get(raw, []))
        cate_lines.append(f"{new} {' '.join(map(str, cats))}")
    with open(os.path.join(args.output_dir, 'ml1m_cate.txt'), 'w') as f:
        f.write('\n'.join(cate_lines) + '\n')

    # ---------- multi-hot vec.npy ----------
    item_vecs = np.zeros((n_items, n_cat), dtype=np.float32)
    for raw, new in item_map.items():
        for g in genres_by_raw.get(raw, []):
            item_vecs[new, genre_map[g]] = 1.0
    np.save(os.path.join(args.output_dir, 'ml1m_vec.npy'), item_vecs)
    print(f"Saved ml1m_vec.npy with shape {item_vecs.shape}")

    # ---------- negatives (99 per test sequence, seed 4444) ----------
    generate_negatives(os.path.join(args.output_dir, 'test-v0.txt'),
                       os.path.join(args.output_dir, 'ML1M-random-sample_size=99-seed=4444.txt'),
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
