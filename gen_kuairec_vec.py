#!/usr/bin/env python3
"""
Generate kuairec_vec.npy from category file (one-hot genre vectors).
Items with multiple categories get 1.0 in each category position.
"""
import numpy as np
import sys
import os

n_items = 10728
n_cats = 31  # categories 0-30

# Build one-hot category vectors
item_vecs = np.zeros((n_items, n_cats), dtype=np.float32)

cate_file = sys.argv[1] if len(sys.argv) > 1 else './KuaiRec_variants/kuairec_first_average/kuairec_cate.txt'
out_file = sys.argv[2] if len(sys.argv) > 2 else './KuaiRec_variants/kuairec_vec.npy'

with open(cate_file) as f:
    for line in f:
        parts = line.strip().split(' ')
        if len(parts) < 2:
            continue
        item_id = int(parts[0])
        cats = [int(c) for c in parts[1:]]
        for c in cats:
            if 0 <= c < n_cats:
                item_vecs[item_id][c] = 1.0

np.save(out_file, item_vecs)
print(f"Saved {out_file}: shape={item_vecs.shape}")
print(f"Items with 0 categories: {np.sum(item_vecs.sum(axis=1) == 0)}")
print(f"Items with 1 category:    {np.sum(item_vecs.sum(axis=1) == 1)}")
print(f"Items with 2+ categories: {np.sum(item_vecs.sum(axis=1) >= 2)}")

# Quick ILD test
from sklearn.metrics.pairwise import pairwise_distances
d = pairwise_distances(item_vecs[:10], metric='euclidean')
print(f"\nSample ILD (10 items, raw): {np.sum(d)/(10*9):.4f}")
print(f"Distance range: {d[d>0].min():.4f} - {d.max():.4f}")
