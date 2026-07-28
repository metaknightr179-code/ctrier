#!/usr/bin/env python3
"""
Convert KuaiRec dataset format to Yelp dataset format for TRIER model.

KuaiRec format:
  - big_matrix.csv / small_matrix.csv: CSV with columns user_id, video_id, timestamp, 
    play_duration, video_duration, watch_ratio, etc.
  - item_categories.csv: CSV with video_id and list of categories (feat column)

Yelp format (target):
  - train-v0.txt, test-v0.txt, valid-v0.txt: Each line is "user_id→item1 item2 ... itemN"
  - yelp_cate.txt: Each line is "item_id→category1 category2 ..."

Key Features:
  - Engagement signal filtering using watch_ratio (default: >= 0.1)
  - Two filtering methods:
    1. Individual filtering: Keep interactions where watch_ratio >= threshold
    2. Average filtering: Calculate avg watch_ratio per user-video pair, keep pairs with avg >= threshold
  - Deduplication of repeated user-video interactions (keep highest watch_ratio)
  - Statistics reporting for quality control

Usage:
  python convert_kuairec_to_yelp.py --input_dir /path/to/kuairec/data --output_dir ./KuaiRec
"""

import argparse
import os
import csv
import json
from collections import defaultdict


def convert_interactions(input_csv, output_train, output_valid, output_test, 
                         min_watch_ratio=0.1, use_avg_watch_ratio=False, min_item_users=0):
    """
    Convert KuaiRec interaction CSV to Yelp-style sequence files.
    
    Filtering methods available:
    - Individual filtering (default): Keep interactions where watch_ratio >= min_watch_ratio
    - Average filtering (use_avg_watch_ratio=True): Calculate average watch_ratio per user-video pair,
      keep only pairs where avg_watch_ratio >= min_watch_ratio
    - Item popularity filtering (min_item_users > 0): Remove items with fewer than min_item_users unique viewers
    
    Steps:
    1. Read interactions with engagement signals
    2. (Optional) Filter items by minimum unique users
    3. Filter interactions based on watch_ratio (individual or average)
    4. Deduplicate repeated user-video interactions (keep highest watch_ratio)
    5. Sort by user_id and timestamp
    6. Group interactions by user to create sequences
    7. Split sequences into train/valid/test (time-based)
    8. Write to output files in Yelp format
    
    Args:
        input_csv: Path to KuaiRec interaction CSV
        output_train: Path to output train file
        output_valid: Path to output valid file
        output_test: Path to output test file
        min_watch_ratio: Minimum watch_ratio threshold (default: 0.1)
        use_avg_watch_ratio: If True, use average watch_ratio per user-video pair (default: False)
        min_item_users: Minimum number of unique users that must watch an item (default: 0, no filtering)
    """
    print(f"Reading interactions from {input_csv}...")
    if use_avg_watch_ratio:
        print(f"Using average watch_ratio filtering with threshold >= {min_watch_ratio}")
    else:
        print(f"Using individual watch_ratio filtering with threshold >= {min_watch_ratio}")
    if min_item_users > 0:
        print(f"Filtering items with fewer than {min_item_users} unique viewers")
    
    # Read all interactions with engagement signals
    all_interactions = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = int(row['user_id'])
            video_id = int(row['video_id'])
            timestamp = float(row['timestamp'])
            watch_ratio = float(row['watch_ratio']) if row['watch_ratio'] else 0.0
            play_duration = float(row['play_duration']) if row['play_duration'] else 0.0
            video_duration = float(row['video_duration']) if row['video_duration'] else 0.0
            
            all_interactions.append({
                'user_id': user_id,
                'video_id': video_id,
                'timestamp': timestamp,
                'watch_ratio': watch_ratio,
                'play_duration': play_duration,
                'video_duration': video_duration
            })
    
    print(f"Total raw interactions: {len(all_interactions)}")
    
    # Filter items by minimum unique users (popularity filtering)
    if min_item_users > 0:
        # Count unique users per item
        item_user_counts = defaultdict(set)
        for i in all_interactions:
            item_user_counts[i['video_id']].add(i['user_id'])
        
        print(f"Total unique items before filtering: {len(item_user_counts)}")
        
        # Get items with sufficient users
        popular_items = {
            item_id for item_id, users in item_user_counts.items() 
            if len(users) >= min_item_users
        }
        
        print(f"Items with >= {min_item_users} unique viewers: {len(popular_items)} ({len(popular_items)/len(item_user_counts)*100:.1f}%)")
        
        # Filter out interactions with unpopular items
        all_interactions = [
            i for i in all_interactions 
            if i['video_id'] in popular_items
        ]
        
        print(f"Interactions after popularity filter: {len(all_interactions)}")
    
    if use_avg_watch_ratio:
        # METHOD: Average watch_ratio filtering
        # Group interactions by user-video pair
        print("Calculating average watch_ratio per user-video pair...")
        user_video_groups = defaultdict(list)
        for i in all_interactions:
            key = (i['user_id'], i['video_id'])
            user_video_groups[key].append(i)
        
        print(f"Total user-video pairs: {len(user_video_groups)}")
        
        # Calculate average watch_ratio for each pair and filter
        filtered_pairs = []
        for key, interactions in user_video_groups.items():
            avg_watch_ratio = sum(i['watch_ratio'] for i in interactions) / len(interactions)
            if avg_watch_ratio >= min_watch_ratio:
                # For retained pairs, keep the interaction with highest watch_ratio
                best_interaction = max(interactions, key=lambda i: (i['watch_ratio'], i['timestamp']))
                filtered_pairs.append(best_interaction)
        
        print(f"User-video pairs with avg_watch_ratio >= {min_watch_ratio}: {len(filtered_pairs)} ({len(filtered_pairs)/len(user_video_groups)*100:.1f}%)")
        deduplicated = filtered_pairs
    else:
        # METHOD: Individual watch_ratio filtering
        # Filter by watch_ratio (engagement signal)
        filtered_interactions = [
            i for i in all_interactions 
            if i['watch_ratio'] >= min_watch_ratio
        ]
        print(f"Interactions after watch_ratio filter: {len(filtered_interactions)} ({len(filtered_interactions)/len(all_interactions)*100:.1f}%)")
        
        # Deduplicate: for each user-video pair, keep only the interaction with highest watch_ratio
        print("Deduplicating user-video interactions...")
        filtered_interactions.sort(key=lambda i: (i['user_id'], i['video_id'], -i['watch_ratio'], -i['timestamp']))
        
        # Keep only the first occurrence for each user-video pair
        deduplicated = []
        seen  = set()
        for i in filtered_interactions:
            key = (i['user_id'], i['video_id'])
            if key not in seen:
                seen.add(key)
                deduplicated.append(i)
        
        print(f"Interactions after deduplication: {len(deduplicated)} ({len(deduplicated)/len(filtered_interactions)*100:.1f}% of filtered)")
    
    # Sort by user_id, then timestamp for sequence creation
    deduplicated.sort(key=lambda x: (x['user_id'], x['timestamp']))
    
    # Group by user to create sequences
    user_sequences = defaultdict(list)
    for interaction in deduplicated:
        user_sequences[interaction['user_id']].append(interaction['video_id'])
    
    print(f"Found {len(user_sequences)} users with sequences")
    
    # Calculate sequence length statistics
    seq_lengths = [len(items) for items in user_sequences.values()]
    if seq_lengths:
        print(f"Sequence length statistics:")
        print(f"  - Min: {min(seq_lengths)}")
        print(f"  - Max: {max(seq_lengths)}")
        print(f"  - Average: {sum(seq_lengths)/len(seq_lengths):.1f}")
        print(f"  - Median: {sorted(seq_lengths)[len(seq_lengths)//2]}")
    
    # Convert sequences to Yelp format lines
    all_lines = []
    for user_id, items in user_sequences.items():
        if len(items) >= 2:  # Need at least 2 items for train/test split
            items_str = ' '.join(map(str, items))
            all_lines.append(f"{user_id}→{items_str}")
    
    print(f"Generated {len(all_lines)} sequences (users with >=2 items)")
    
    # Split into train/valid/test (time-based split per user)
    # We'll use the last items for validation and test
    train_lines = []
    valid_lines = []
    test_lines = []
    
    for line in all_lines:
        parts = line.split('→')
        user_id = parts[0]
        items = parts[1].split()
        
        # Time-based split: use all but last 2 items for train
        # Second-to-last for validation, last for test
        if len(items) >= 3:
            train_items = items[:-2]
            valid_items = items[:-1]
            test_items = items
            
            train_lines.append(f"{user_id}→{' '.join(train_items)}")
            valid_lines.append(f"{user_id}→{' '.join(valid_items)}")
            test_lines.append(f"{user_id}→{' '.join(test_items)}")
        elif len(items) == 2:
            # Only 2 items: train with first, valid/test with both
            train_lines.append(f"{user_id}→{items[0]}")
            valid_lines.append(line)
            test_lines.append(line)
    
    # Write output files
    print(f"Writing {len(train_lines)} train sequences to {output_train}")
    with open(output_train, 'w') as f:
        f.write('\n'.join(train_lines) + '\n')
    
    print(f"Writing {len(valid_lines)} valid sequences to {output_valid}")
    with open(output_valid, 'w') as f:
        f.write('\n'.join(valid_lines) + '\n')
    
    print(f"Writing {len(test_lines)} test sequences to {output_test}")
    with open(output_test, 'w') as f:
        f.write('\n'.join(test_lines) + '\n')
    
    return len(user_sequences), len(all_lines)


def convert_categories(input_csv, output_cate):
    """
    Convert KuaiRec item_categories.csv to Yelp-style category file.
    
    KuaiRec format: video_id, feat (list like "[8]" or "[27, 9]")
    Yelp format: item_id→category1 category2 ...
    """
    print(f"Reading categories from {input_csv}...")
    
    cate_lines = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = int(row['video_id'])
            feat_str = row['feat']
            
            # Parse the feat list (format: "[8]" or "[27, 9]")
            try:
                categories = json.loads(feat_str)
                if isinstance(categories, list):
                    categories_str = ' '.join(map(str, categories))
                    cate_lines.append(f"{video_id}→{categories_str}")
            except json.JSONDecodeError:
                # Handle cases where parsing fails
                print(f"Warning: Could not parse categories for video {video_id}: {feat_str}")
    
    print(f"Writing {len(cate_lines)} category entries to {output_cate}")
    with open(output_cate, 'w') as f:
        f.write('\n'.join(cate_lines) + '\n')
    
    return len(cate_lines)


def generate_negatives(test_file, output_neg_file, item_num):
    """
    Generate negative samples file for testing.
    Yelp format: each line has 99 negative item IDs.
    """
    print(f"Generating negatives for {test_file}...")
    
    import random
    
    negatives = []
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('→')
            items = list(map(int, parts[1].split()))
            
            # Get all items in the sequence (to avoid sampling as negatives)
            seq_items = set(items)
            
            # Generate 99 random negative items
            neg_items = []
            while len(neg_items) < 99:
                neg = random.randint(1, item_num - 1)
                if neg not in seq_items:
                    neg_items.append(neg)
            
            negatives.append(' '.join(map(str, neg_items)))
    
    print(f"Writing {len(negatives)} negative samples to {output_neg_file}")
    with open(output_neg_file, 'w') as f:
        f.write('\n'.join(negatives) + '\n')
    
    return len(negatives)


def count_unique_items(input_csv):
    """Count unique items (videos) in the interaction file."""
    items = set()
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.add(int(row['video_id']))
    return len(items)


def main():
    parser = argparse.ArgumentParser(description='Convert KuaiRec to Yelp format')
    parser.add_argument('--input_dir', required=True, help='Path to KuaiRec data directory')
    parser.add_argument('--output_dir', default='./KuaiRec', help='Output directory')
    parser.add_argument('--interaction_file', default='big_matrix.csv', 
                        help='Interaction file name (big_matrix.csv or small_matrix.csv)')
    parser.add_argument('--min_watch_ratio', type=float, default=0.1, 
                        help='Minimum watch_ratio to include interaction (default: 0.1). '
                             'Use 0.0 to include all interactions.')
    parser.add_argument('--use_avg_watch_ratio', action='store_true', 
                        help='If set, calculate average watch_ratio per user-video pair '
                             'and filter based on the average instead of individual values.')
    parser.add_argument('--min_item_users', type=int, default=0, 
                        help='Minimum number of unique users that must watch an item '
                             '(default: 0, no filtering). Use to remove rare items.')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths
    interaction_path = os.path.join(args.input_dir, args.interaction_file)
    categories_path = os.path.join(args.input_dir, 'item_categories.csv')
    
    output_train = os.path.join(args.output_dir, 'train-v0.txt')
    output_valid = os.path.join(args.output_dir, 'valid-v0.txt')
    output_test = os.path.join(args.output_dir, 'test-v0.txt')
    output_cate = os.path.join(args.output_dir, 'kuairec_cate.txt')
    output_neg = os.path.join(args.output_dir, 'KuaiRec-random-sample_size=99-seed=4444.txt')
    
    # Check input files exist
    if not os.path.exists(interaction_path):
        print(f"Error: Interaction file not found at {interaction_path}")
        return
    
    if not os.path.exists(categories_path):
        print(f"Error: Categories file not found at {categories_path}")
        return
    
    # Count items for negative sampling
    item_num = count_unique_items(interaction_path)
    print(f"Total unique items: {item_num}")
    
    # Convert interactions (with engagement signal filtering)
    convert_interactions(interaction_path, output_train, output_valid, output_test, 
                         min_watch_ratio=args.min_watch_ratio,
                         use_avg_watch_ratio=args.use_avg_watch_ratio,
                         min_item_users=args.min_item_users)
    
    # Convert categories
    convert_categories(categories_path, output_cate)
    
    # Generate negatives
    generate_negatives(output_test, output_neg, item_num)
    
    print("\nConversion complete!")
    print(f"Output files written to: {args.output_dir}")


if __name__ == '__main__':
    main()
