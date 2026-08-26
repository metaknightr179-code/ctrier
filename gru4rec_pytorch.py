#!/usr/bin/env python3
"""
GRU4Rec PyTorch implementation for Kuairec baseline comparison.
Uses BPTT (all positions predict next item) and batch evaluation.
Computes the same metrics as TRIER (recall, MRR, NDCG, ILD, CS, CC) for fair comparison.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script import evaluate_function_with_full, get_metrics_full, get_cates_map


class GRU4RecModel(nn.Module):
    def __init__(self, item_num, embedding_dim=64, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.item_num = item_num
        self.embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, item_num + 1)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, _ = self.gru(embedded)
        output = self.dropout(output)
        logits = self.output_proj(output)  # (B, L, item_num+1)
        return logits


class SeqDataset(Dataset):
    """Dataset for BPTT training: each position predicts the next item."""
    def __init__(self, data_file, maxlen=50):
        self.data = []
        self.maxlen = maxlen
        with open(data_file, 'r') as f:
            for line in f:
                items = [int(x) for x in line.strip().split()]
                if len(items) >= 2:
                    self.data.append(items)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx][-self.maxlen:]
        # Input: seq[:-1], Target: seq[1:]  (shift by 1)
        input_seq = seq[:-1]
        target_seq = seq[1:]
        # Pad to maxlen-1
        pad_len = (self.maxlen - 1) - len(input_seq)
        padded_input = [0] * pad_len + input_seq
        padded_target = [0] * pad_len + target_seq
        return (
            torch.tensor(padded_input, dtype=torch.long),
            torch.tensor(padded_target, dtype=torch.long),
        )


class EvalDataset(Dataset):
    """Dataset for evaluation: predict last item from preceding sequence."""
    def __init__(self, data_file, maxlen=50):
        self.data = []
        self.maxlen = maxlen
        with open(data_file, 'r') as f:
            for line in f:
                items = [int(x) for x in line.strip().split()]
                if len(items) >= 2:
                    self.data.append(items)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        input_seq = seq[:-1][-(self.maxlen - 1):]
        target = seq[-1]
        padded = [0] * ((self.maxlen - 1) - len(input_seq)) + input_seq
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def train_gru4rec(train_file, item_num, epochs, batch_size, lr, maxlen, ckpt_dir, device):
    """Train GRU4Rec with BPTT (every position predicts next item)."""
    dataset = SeqDataset(train_file, maxlen)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = GRU4RecModel(item_num, embedding_dim=64, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'gru4rec_best.pth')
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0

        for batch_input, batch_target in dataloader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            logits = model(batch_input)  # (B, L, item_num+1)
            # Flatten for cross entropy: (B*L, item_num+1) vs (B*L,)
            loss = criterion(logits.reshape(-1, item_num + 1), batch_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

        print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}', flush=True)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model


def evaluate_gru4rec(model, test_file, item_num, maxlen, batch_size, cat_map, cat_num, item2vec, device):
    """Evaluate GRU4Rec in batches with full metrics (recall, MRR, NDCG, ILD, CS, CC)."""
    dataset = EvalDataset(test_file, maxlen)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    total_result = []

    with torch.no_grad():
        for batch_input, batch_target in dataloader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            logits = model(batch_input)  # (B, L, item_num+1)
            # Use last non-pad position for prediction
            last_logits = logits[:, -1, :]  # (B, item_num+1)
            last_logits[:, 0] = float('-inf')  # exclude padding

            _, rec_list = last_logits.topk(k=20, dim=-1)  # (B, 20)

            result = evaluate_function_with_full(
                batch_target, rec_list,
                cat_map=cat_map, cat_num=cat_num, item2vec=item2vec
            )
            total_result.extend(result)

    # Aggregate metrics
    metrics = {}
    for name in ['recall@5_f', 'recall@10_f', 'recall@20_f',
                 'mrr@5_f', 'mrr@10_f', 'mrr@20_f',
                 'ndcg@5_f', 'ndcg@10_f', 'ndcg@20_f',
                 'ILD@5', 'ILD@10', 'ILD@20',
                 'CS@5', 'CS@10', 'CS@20',
                 'CC@5', 'CC@10', 'CC@20']:
        metrics[name] = get_metrics_full(name, total_result)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='GRU4Rec baseline')
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--item_num', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--cat', type=str, default=None, help='Category file')
    parser.add_argument('--n_cat', type=int, default=0, help='Number of categories')
    parser.add_argument('--vec', type=str, default=None, help='Item2vec .npy file')
    parser.add_argument('--output', default='gru4rec_results.txt')
    parser.add_argument('--ckpt_dir', default='./save_gru4rec')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only evaluate saved checkpoint')
    parser.add_argument('--ckpt_path', default=None, help='Path to checkpoint file for eval_only mode')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load category mapping
    cat_map = None
    if args.cat and os.path.exists(args.cat):
        cat_map = get_cates_map(args.cat)
        print(f'Loaded category mapping from {args.cat}')

    # Load item2vec for diversity metrics
    item2vec = None
    for vec_path in [args.vec, './KuaiRec_variants/kuairec_vec.npy', './kuairec_vec.npy']:
        if vec_path and os.path.exists(vec_path):
            item2vec = torch.tensor(np.load(vec_path))
            print(f'Loaded item embeddings from {vec_path}')
            break
    if item2vec is not None and torch.cuda.is_available():
        item2vec = item2vec.to(device)

    if args.eval_only:
        # Eval-only mode: load checkpoint and evaluate
        ckpt_path = args.ckpt_path or os.path.join(args.ckpt_dir, 'gru4rec_best.pth')
        if not os.path.exists(ckpt_path):
            print(f'ERROR: Checkpoint not found: {ckpt_path}')
            sys.exit(1)
        print(f'Eval-only mode: loading checkpoint from {ckpt_path}')
        model = GRU4RecModel(args.item_num, embedding_dim=64, hidden_dim=64).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        train_time = 0.0
    else:
        # Train
        start_time = time.time()
        model = train_gru4rec(args.train_file, args.item_num, args.epochs,
                              args.batch_size, args.lr, args.maxlen, args.ckpt_dir, device)
        train_time = time.time() - start_time

    # Evaluate
    results = evaluate_gru4rec(model, args.test_file, args.item_num, args.maxlen,
                                args.batch_size, cat_map, args.n_cat, item2vec, device)

    # Print results
    print('\n' + '=' * 50)
    print('GRU4Rec Results')
    print('=' * 50)
    for k in [5, 10, 20]:
        print(f'Recall@{k}: {results[f"recall@{k}_f"]:.4f}')
        print(f'MRR@{k}:    {results[f"mrr@{k}_f"]:.4f}')
        print(f'NDCG@{k}:   {results[f"ndcg@{k}_f"]:.4f}')
    for k in [5, 10, 20]:
        print(f'ILD@{k}:    {results[f"ILD@{k}"]:.4f}')
        print(f'CS@{k}:     {results[f"CS@{k}"]:.4f}')
        print(f'CC@{k}:     {results[f"CC@{k}"]:.4f}')

    # Save results
    with open(args.output, 'w') as f:
        f.write(f'GRU4Rec Results\n')
        f.write(f'Training time: {train_time:.1f}s\n')
        f.write(f'Epochs: {args.epochs}\n\n')
        for k in [5, 10, 20]:
            f.write(f'Recall@{k}: {results[f"recall@{k}_f"]:.4f}\n')
            f.write(f'MRR@{k}:    {results[f"mrr@{k}_f"]:.4f}\n')
            f.write(f'NDCG@{k}:   {results[f"ndcg@{k}_f"]:.4f}\n')
        for k in [5, 10, 20]:
            f.write(f'ILD@{k}:    {results[f"ILD@{k}"]:.4f}\n')
            f.write(f'CS@{k}:     {results[f"CS@{k}"]:.4f}\n')
            f.write(f'CC@{k}:     {results[f"CC@{k}"]:.4f}\n')

    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
