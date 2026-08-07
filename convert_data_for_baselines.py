#!/usr/bin/env python3
"""
Convert Kuairec data to baseline model formats (GRU4Rec, SASRec).
"""
import argparse
import os


def convert_to_gru4rec(input_dir, output_dir):
    """
    Convert to GRU4Rec format: tab-separated with columns SessionId, ItemId, Time.
    Each line in Kuairec data is a user's sequence of item IDs (space-separated).
    """
    for split_name in ['train', 'test']:
        input_file = os.path.join(input_dir, f'{split_name}-v0.txt')
        output_file = os.path.join(output_dir, f'gru4rec_{split_name}.tsv')
        
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            fout.write('SessionId\tItemId\tTime\n')
            for user_id, line in enumerate(fin, start=1):
                items = line.strip().split()
                for seq_pos, item_id in enumerate(items):
                    fout.write(f'{user_id}\t{item_id}\t{seq_pos}\n')
        
        print(f'GRU4Rec {split_name}: {os.path.getsize(output_file)} bytes')


def convert_to_sasrec(input_dir, output_dir):
    """
    Convert to SASRec format: space-separated user_id item_id, one per line.
    SASRec expects a single file with all interactions.
    """
    output_file = os.path.join(output_dir, 'sasrec_data.txt')
    
    all_users = {}
    max_user_id = 0
    
    for split_idx, split_name in enumerate(['train', 'valid', 'test']):
        input_file = os.path.join(input_dir, f'{split_name}-v0.txt')
        with open(input_file, 'r') as fin:
            for line in fin:
                user_items = line.strip().split()
                if user_items:
                    all_users[max_user_id] = user_items
                    max_user_id += 1
    
    with open(output_file, 'w') as fout:
        for user_id, items in all_users.items():
            for item_id in items:
                fout.write(f'{user_id + 1} {item_id}\n')
    
    print(f'SASRec data: {os.path.getsize(output_file)} bytes, {max_user_id} users')


def main():
    parser = argparse.ArgumentParser(description='Convert data for baseline models')
    parser.add_argument('--input_dir', required=True, help='Input directory with Kuairec data')
    parser.add_argument('--output_dir', required=True, help='Output directory for converted data')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('Converting data for GRU4Rec...')
    convert_to_gru4rec(args.input_dir, args.output_dir)
    
    print('Converting data for SASRec...')
    convert_to_sasrec(args.input_dir, args.output_dir)
    
    print('Done!')


if __name__ == '__main__':
    main()
