#!/usr/bin/env python3
"""
Generate kuairec_music.txt from KuaiRec item_daily_features.csv.

Each video has one background-music id; music_id is stable across dates
(we take the modal value per video). Raw music ids are remapped to a compact
1..N range (0 = padding/unknown/no-music, mirroring the type/author convention).

Output format (same as kuairec_cate.txt / kuairec_author.txt, one id per item):
    <item_id> <music_remapped>

Usage:
    python3 gen_kuairec_music.py \
        [input_csv=./KuaiRec/data/item_daily_features.csv] \
        [out_file=./KuaiRec_variants/kuairec_music.txt]
"""
import csv
import sys
from collections import Counter, defaultdict

input_csv = sys.argv[1] if len(sys.argv) > 1 else './KuaiRec/data/item_daily_features.csv'
out_file = sys.argv[2] if len(sys.argv) > 2 else './KuaiRec_variants/kuairec_music.txt'

# video_id -> Counter of music_id (modal guards against daily noise)
video_music = defaultdict(Counter)
with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = row['video_id']
        mid = row.get('music_id', '')
        if vid == '' or mid == '':
            continue
        mid = int(mid)
        if mid == 0:
            continue  # 0 = no music / unknown -> padding
        video_music[int(vid)][mid] += 1

# Modal music per video (only videos with at least one nonzero music id)
item_music = {vid: cnt.most_common(1)[0][0] for vid, cnt in video_music.items()}

# Compact remap of raw music ids -> 1..N (0 reserved for padding/no-music)
unique_music = sorted(set(item_music.values()))
music_map = {mid: i + 1 for i, mid in enumerate(unique_music)}
n_music = len(unique_music)

n_items = 10728  # KuaiRec catalog size (matches -n 10728)
lines = []
n_covered = 0
for item_id in range(n_items):
    raw = item_music.get(item_id)
    if raw is None:
        continue  # no music -> padding (0)
    lines.append(f"{item_id} {music_map[raw]}")
    n_covered += 1

with open(out_file, 'w') as f:
    f.write('\n'.join(lines) + '\n')

raw_counts = Counter(item_music.values())
print(f"Saved {out_file}")
print(f"Unique music ids: {n_music}  -> -n_music {n_music}")
print(f"Items with music: {n_covered}/{n_items} ({n_covered/n_items*100:.1f}%)")
print(f"Items per music: mean={n_covered/n_music:.2f}, "
      f"max={max(raw_counts.values())}")
