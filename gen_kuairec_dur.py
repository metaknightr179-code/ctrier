#!/usr/bin/env python3
"""
Generate kuairec_dur.txt from KuaiRec item_daily_features.csv.

video_duration is a stable per-video content attribute in MILLISECONDS.
Daily rows can disagree slightly, so we take the per-video MEDIAN, then
bucketize into 8 ordinal short-video length buckets (ids 1..8;
0 = padding/unknown for the ~2.7% of items without a duration):

    1: <3s        5: 10-15s
    2: 3-5s       6: 15-30s
    3: 5-7s       7: 30-60s
    4: 7-10s      8: >60s

Output format (same as kuairec_author.txt):
    <item_id> <duration_bucket>

Usage:
    python3 gen_kuairec_dur.py \
        [input_csv=./KuaiRec/data/item_daily_features.csv] \
        [out_file=./KuaiRec_variants/kuairec_dur.txt]
"""
import csv
import sys
from collections import defaultdict
import numpy as np

input_csv = sys.argv[1] if len(sys.argv) > 1 else './KuaiRec/data/item_daily_features.csv'
out_file = sys.argv[2] if len(sys.argv) > 2 else './KuaiRec_variants/kuairec_dur.txt'

# Bucket right-edges in milliseconds; bucket id = first edge d <= edge's index+1
EDGES_MS = [3000, 5000, 7000, 10000, 15000, 30000, 60000]
N_DUR = len(EDGES_MS) + 1  # 8 buckets


def bucket_id(duration_ms):
    for i, edge in enumerate(EDGES_MS):
        if duration_ms <= edge:
            return i + 1
    return N_DUR


# video_id -> list of daily duration observations
video_durs = defaultdict(list)
with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = row['video_id']
        d = row.get('video_duration', '')
        if vid == '' or d == '':
            continue
        try:
            d = float(d)
        except ValueError:
            continue
        if d > 0:
            video_durs[int(vid)].append(d)

# Median duration per video -> bucket
item_bucket = {vid: bucket_id(float(np.median(ds)))
               for vid, ds in video_durs.items()}

n_items = 10728  # KuaiRec catalog size (matches -n 10728)
lines = []
bucket_counts = [0] * (N_DUR + 1)
for item_id in range(n_items):
    b = item_bucket.get(item_id)
    if b is None:
        continue  # unknown -> padding (0)
    lines.append(f"{item_id} {b}")
    bucket_counts[b] += 1

with open(out_file, 'w') as f:
    f.write('\n'.join(lines) + '\n')

n_covered = len(lines)
print(f"Saved {out_file}")
print(f"Duration buckets: {N_DUR}  -> -n_dur {N_DUR}")
print(f"Items with duration: {n_covered}/{n_items} ({n_covered/n_items*100:.1f}%)")
labels = ["<3s", "3-5s", "5-7s", "7-10s", "10-15s", "15-30s", "30-60s", ">60s"]
for b in range(1, N_DUR + 1):
    print(f"  bucket {b} ({labels[b-1]:>7}): {bucket_counts[b]:6d} "
          f"({bucket_counts[b]/n_items*100:5.1f}%)")
