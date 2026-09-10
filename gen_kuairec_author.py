#!/usr/bin/env python3
"""
Generate kuairec_author.txt from KuaiRec item_daily_features.csv.

Each video has one author; author_id is stable across dates (we take the
modal value per video). Raw author ids are remapped to a compact 1..N range
(0 = padding/unknown, mirroring the type-embedding convention).

Output format (same as kuairec_cate.txt, one id per item):
    <item_id> <author_remapped>

Usage:
    python3 gen_kuairec_author.py \
        [input_csv=./KuaiRec/data/item_daily_features.csv] \
        [out_file=./KuaiRec_variants/kuairec_author.txt]
"""
import csv
import sys
from collections import Counter, defaultdict

input_csv = sys.argv[1] if len(sys.argv) > 1 else './KuaiRec/data/item_daily_features.csv'
out_file = sys.argv[2] if len(sys.argv) > 2 else './KuaiRec_variants/kuairec_author.txt'

# video_id -> Counter of author_id (author should be stable; modal guards against noise)
video_authors = defaultdict(Counter)
with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = row['video_id']
        aid = row.get('author_id', '')
        if vid == '' or aid == '':
            continue
        video_authors[int(vid)][int(aid)] += 1

# Modal author per video
item_author = {vid: cnt.most_common(1)[0][0] for vid, cnt in video_authors.items()}

# Compact remap of raw author ids -> 1..N (0 reserved for padding/unknown)
unique_authors = sorted(set(item_author.values()))
author_map = {aid: i + 1 for i, aid in enumerate(unique_authors)}
n_author = len(unique_authors)

n_items = 10728  # KuaiRec catalog size (matches -n 10728)
lines = []
n_covered = 0
for item_id in range(n_items):
    raw = item_author.get(item_id)
    if raw is None:
        continue  # unknown -> all zeros buffer (padding)
    lines.append(f"{item_id} {author_map[raw]}")
    n_covered += 1

with open(out_file, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Saved {out_file}")
print(f"Unique authors: {n_author}  -> -n_author {n_author}")
print(f"Items with author: {n_covered}/{n_items} ({n_covered/n_items*100:.1f}%)")
print(f"Items per author: mean={n_covered/n_author:.2f}, "
      f"max={max(Counter(item_author.values()).values())}")
