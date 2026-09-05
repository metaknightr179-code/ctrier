#!/usr/bin/env python3
"""
Evaluate RT model using built-in test mode.
"""

import torch
import torch.utils.data as Data
import numpy as np
import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trier_rt import TRIER_RT
from dataset_duorec import TestDataset
from script import *


def analyze_training_curve(log_file):
    """Analyze training loss convergence from log file."""
    print("=" * 60)
    print("TRAINING LOSS ANALYSIS")
    print("=" * 60)
    
    epochs = []
    losses = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'loss' in line:
                parts = line.strip().split()
                epoch = int(parts[1])
                loss = float(parts[3])
                epochs.append(epoch)
                losses.append(loss)
    
    if not losses:
        print("No loss data found!")
        return None
    
    print(f"Total epochs: {len(losses)}")
    print(f"Initial loss: {losses[0]:.4f} (epoch {epochs[0]})")
    print(f"Final loss:   {losses[-1]:.4f} (epoch {epochs[-1]})")
    print(f"Loss reduction: {losses[0] - losses[-1]:.4f} ({(1 - losses[-1]/losses[0])*100:.1f}%)")
    
    # Check for convergence
    if len(losses) >= 10:
        last_10_avg = sum(losses[-10:]) / 10
        prev_10_avg = sum(losses[-20:-10]) / 10 if len(losses) >= 20 else sum(losses[:10]) / 10
        improvement = prev_10_avg - last_10_avg
        print(f"\nLast 10 epochs avg loss: {last_10_avg:.4f}")
        print(f"Previous 10 epochs avg loss: {prev_10_avg:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement/prev_10_avg*100:.2f}%)")
        
        if improvement < 0.01:
            status = "CONVERGED"
        elif improvement < 0.05:
            status = "NEARLY CONVERGED"
        elif improvement < 0.1:
            status = "SLOWLY CONVERGING"
        else:
            status = "STILL LEARNING"
        print(f"Status: {status}")
    
    # Show loss every 10 epochs
    print("\nLoss progression (every 10 epochs):")
    print(f"{'Epoch':<10} {'Loss':<12} {'Delta':<12}")
    print("-" * 34)
    for i in range(0, len(losses), 10):
        print(f"{epochs[i]:<10} {losses[i]:<12.4f} {'-':<12}")
        if i+9 < len(losses):
            delta = losses[i+9] - losses[i]
            print(f"{epochs[i+9]:<10} {losses[i+9]:<12.4f} {delta:<+12.4f}")
    
    return losses


def evaluate_rt_checkpoint(model_path, test_file, neg_file, item_num, batch_size, device, item2vec=None):
    """Evaluate a single RT model checkpoint."""
    import argparse
    from script import get_args
    import sys as _sys
    
    # Save original argv
    old_argv = _sys.argv.copy()
    _sys.argv = ['script.py']
    args = get_args()
    _sys.argv = old_argv
    args.device = device
    
    hidden_unit = 64
    head_num = 4
    layer_num = 2
    dropout_rate = 0.1
    max_seq_len = 50
    
    # Load model
    model = TRIER_RT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test dataset
    modified_max_seq_len = 72
    dataset = TestDataset(test_file, neg_file, item_num, max_seq_len, modified_max_seq_len)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Evaluate
    total_result = []
    with torch.no_grad():
        for batch in dataloader:
            input_session_ids, targets, negatives, _, _ = batch
            input_session_ids = input_session_ids.to(device)
            targets = targets.to(device)
            
            output = model.test_forward(input_session_ids)
            output = torch.matmul(output, model.item_embedding.weight.T)
            _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
            
            # Pass item2vec for diversity metrics (ILD)
            result = evaluate_function_with_full(targets, rec_list, item2vec=item2vec)
            if result:
                total_result.extend(result)
    
    # Compute metrics
    if not total_result:
        print("  Warning: No results computed!")
        return None
    
    metrics = {}
    for metric in ['recall@5_f', 'recall@10_f', 'recall@20_f', 
                   'mrr@5_f', 'mrr@10_f', 'mrr@20_f',
                   'ndcg@5_f', 'ndcg@10_f', 'ndcg@20_f']:
        metrics[metric] = get_metrics_full(metric, total_result)
    
    # Add ILD metrics if available
    if 'ILD@5' in total_result[0]:
        metrics['ILD@5'] = get_metrics_full('ILD@5', total_result)
        metrics['ILD@10'] = get_metrics_full('ILD@10', total_result)
        metrics['ILD@20'] = get_metrics_full('ILD@20', total_result)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate RT model')
    parser.add_argument('--model_dir', default='./save_rt_yelp', help='Directory with model checkpoints')
    parser.add_argument('--test_file', default='./Yelp/test-v0.txt', help='Test file')
    parser.add_argument('--neg_file', default='./Yelp/Yelp-random-sample_size=99-seed=4444.txt', help='Negative samples file')
    parser.add_argument('--item_vec_file', default='./Yelp/yelp_vec.npy', help='Item embeddings file for diversity metrics')
    parser.add_argument('--item_num', type=int, default=14588, help='Number of items')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze training curve')
    parser.add_argument('--epochs', type=str, default='100', help='Comma-separated epochs to evaluate (e.g., "50,75,100")')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Model directory: {args.model_dir}")
    
    # Load item embeddings for diversity metrics
    item2vec = None
    if os.path.exists(args.item_vec_file):
        print(f"Loading item embeddings from {args.item_vec_file}...")
        item2vec = np.load(args.item_vec_file)
        item2vec = torch.tensor(item2vec)
        item2vec = item2vec.to(device)
        print(f"  Item embeddings shape: {item2vec.shape}")
    else:
        print(f"Warning: Item embeddings not found at {args.item_vec_file}")
        print("  Diversity metrics (ILD) will not be computed.")
    
    # Step 1: Analyze training curve
    log_file = os.path.join(args.model_dir, 'train_result.txt')
    if os.path.exists(log_file):
        losses = analyze_training_curve(log_file)
    else:
        print(f"Warning: {log_file} not found!")
        losses = None
    
    if args.analyze_only:
        return
    
    # Step 2: Find available checkpoints
    model_files = glob.glob(os.path.join(args.model_dir, 'model', 'duorec-*.pth'))
    if not model_files:
        print(f"\nError: No checkpoints found in {args.model_dir}/model/")
        return
    
    # Sort by epoch number
    model_files.sort(key=lambda x: int(x.split('duorec-')[1].split('.pth')[0]))
    
    print(f"\n{'='*60}")
    print(f"RT MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Found {len(model_files)} checkpoints:")
    
    # Parse requested epochs
    if args.epochs == 'all':
        eval_epochs = None  # evaluate all
    else:
        eval_epochs = [int(e.strip()) for e in args.epochs.split(',')]
    
    # Filter model files
    eval_files = []
    for f in model_files:
        epoch = int(f.split('duorec-')[1].split('.pth')[0])
        if eval_epochs is None or epoch in eval_epochs:
            eval_files.append((epoch, f))
    
    if not eval_files:
        # If no match, evaluate all
        eval_files = [(int(f.split('duorec-')[1].split('.pth')[0]), f) for f in model_files]
    
    eval_files.sort(key=lambda x: x[0])
    
    print(f"\nEvaluating {len(eval_files)} checkpoints:")
    for epoch, _ in eval_files:
        print(f"  Epoch {epoch}")
    
    # Step 3: Evaluate each checkpoint
    all_metrics = []
    for epoch, model_file in eval_files:
        print(f"\nEvaluating epoch {epoch}...")
        
        try:
            metrics = evaluate_rt_checkpoint(
                model_file, args.test_file, args.neg_file, 
                args.item_num, args.batch_size, device, item2vec
            )
            if metrics:
                metrics['epoch'] = epoch
                all_metrics.append(metrics)
                
                print(f"  Recall@5:  {metrics['recall@5_f']:.4f}")
                print(f"  Recall@10: {metrics['recall@10_f']:.4f}")
                print(f"  Recall@20: {metrics['recall@20_f']:.4f}")
                print(f"  NDCG@5:    {metrics['ndcg@5_f']:.4f}")
                print(f"  NDCG@10:   {metrics['ndcg@10_f']:.4f}")
                print(f"  NDCG@20:   {metrics['ndcg@20_f']:.4f}")
                print(f"  MRR@5:     {metrics['mrr@5_f']:.4f}")
                print(f"  MRR@10:    {metrics['mrr@10_f']:.4f}")
                print(f"  MRR@20:    {metrics['mrr@20_f']:.4f}")
                if 'ILD@5' in metrics:
                    print(f"  ILD@5:     {metrics['ILD@5']:.4f}")
                    print(f"  ILD@10:    {metrics['ILD@10']:.4f}")
                    print(f"  ILD@20:    {metrics['ILD@20']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Summary comparison
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"MODEL PROGRESSION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n{'Epoch':<10} {'Recall@20':<15} {'NDCG@20':<15} {'MRR@20':<15}")
        print("-" * 55)
        for m in all_metrics:
            print(f"{m['epoch']:<10} {m['recall@20_f']:<15.4f} {m['ndcg@20_f']:<15.4f} {m['mrr@20_f']:<15.4f}")
        
        # Find best model by NDCG@20
        all_metrics.sort(key=lambda x: x['ndcg@20_f'], reverse=True)
        best = all_metrics[0]
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL (Epoch {best['epoch']})")
        print(f"{'='*60}")
        
        paper_metrics = {
            'ndcg@20_f': 0.0444, 'ndcg@10_f': 0.0330, 'ndcg@5_f': 0.0244,
            'recall@20_f': 0.1245, 'recall@10_f': 0.0781, 'recall@5_f': 0.0496,
        }
        
        print(f"\n{'Metric':<15} {'Our Model':<15} {'Paper':<15} {'Gap':<15}")
        print("-" * 60)
        for metric, paper_val in paper_metrics.items():
            if metric in best:
                val = best[metric]
                gap = val - paper_val
                print(f"{metric:<15} {val:<15.4f} {paper_val:<15.4f} {gap:<+15.4f}")


if __name__ == '__main__':
    main()