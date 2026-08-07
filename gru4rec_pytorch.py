#!/usr/bin/env python3
"""
GRU4Rec PyTorch implementation for Kuairec baseline comparison.
Optimized for efficiency.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4RecModel(nn.Module):
    def __init__(self, item_num, embedding_dim=100, hidden_dim=100):
        super().__init__()
        self.item_num = item_num
        self.embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, item_num + 1)
        
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, _ = self.gru(embedded)
        logits = self.output_proj(output[:, -1, :])
        return logits


def load_sequences(data_file):
    """Load user sequences from file."""
    sequences = []
    with open(data_file, 'r') as f:
        for line in f:
            items = [int(x) for x in line.strip().split()]
            if len(items) >= 2:
                sequences.append(items)
    return sequences


def create_batches(sequences, batch_size, maxlen=50):
    """Create training batches from sequences."""
    inputs = []
    targets = []
    
    for seq in sequences:
        seq = seq[-maxlen:]
        # Use last item as target, rest as input
        if len(seq) >= 2:
            input_seq = seq[:-1]
            target = seq[-1]
            # Pad input
            padded = [0] * (maxlen - len(input_seq)) + input_seq
            inputs.append(padded)
            targets.append(target)
    
    # Create batches
    batches = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = torch.LongTensor(inputs[i:i+batch_size])
        batch_targets = torch.LongTensor(targets[i:i+batch_size])
        batches.append((batch_inputs, batch_targets))
    
    return batches


def train_gru4rec(train_file, test_file, item_num, epochs=20, batch_size=64, lr=0.001, device='cuda'):
    """Train GRU4Rec model."""
    print(f'Loading training data...')
    train_sequences = load_sequences(train_file)
    print(f'Loaded {len(train_sequences)} valid sequences')
    
    print('Creating training batches...')
    train_batches = create_batches(train_sequences, batch_size)
    print(f'Created {len(train_batches)} batches')
    
    # Model
    model = GRU4RecModel(item_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        # Shuffle batches
        np.random.shuffle(train_batches)
        
        for batch_idx, (batch_input, batch_target) in enumerate(train_batches):
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            logits = model(batch_input)
            loss = criterion(logits, batch_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_batches)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'gru4rec_best.pth')
        
        print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f} (ep {best_epoch})')
    
    # Load best model
    if os.path.exists('gru4rec_best.pth'):
        model.load_state_dict(torch.load('gru4rec_best.pth', weights_only=True))
    model.eval()
    
    return model


def evaluate_gru4rec(model, test_file, item_num, device='cuda'):
    """Evaluate GRU4Rec model."""
    print(f'\nEvaluating...')
    test_sequences = load_sequences(test_file)
    
    model.eval()
    all_recall = {5: [], 10: [], 20: []}
    all_mrr = {5: [], 10: [], 20: []}
    all_ndcg = {5: [], 10: [], 20: []}
    
    with torch.no_grad():
        for seq_idx, seq in enumerate(test_sequences):
            if len(seq) < 2:
                continue
            
            input_seq = seq[:-1][-49:]
            target = seq[-1]
            
            padded = [0] * (50 - len(input_seq)) + input_seq
            input_tensor = torch.LongTensor([padded]).to(device)
            
            logits = model(input_tensor)
            logits[0, 0] = float('-inf')  # Exclude padding
            
            ranked = logits[0].argsort(descending=True).cpu().numpy()
            
            for k in [5, 10, 20]:
                hit = target in ranked[:k]
                all_recall[k].append(1.0 if hit else 0.0)
                
                if hit:
                    rank = np.where(ranked[:k] == target)[0][0] + 1
                    all_mrr[k].append(1.0 / rank)
                    all_ndcg[k].append(1.0 / np.log2(rank + 1))
                else:
                    all_mrr[k].append(0.0)
                    all_ndcg[k].append(0.0)
            
            if (seq_idx + 1) % 1000 == 0:
                print(f'  Evaluated {seq_idx + 1}/{len(test_sequences)}...')
    
    results = {}
    for k in [5, 10, 20]:
        results[f'recall@{k}'] = np.mean(all_recall[k])
        results[f'mrr@{k}'] = np.mean(all_mrr[k])
        results[f'ndcg@{k}'] = np.mean(all_ndcg[k])
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--item_num', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', default='gru4rec_results.txt')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    start_time = time.time()
    model = train_gru4rec(args.train_file, args.test_file, args.item_num,
                         epochs=args.epochs, batch_size=args.batch_size, 
                         lr=args.lr, device=device)
    train_time = time.time() - start_time
    
    results = evaluate_gru4rec(model, args.test_file, args.item_num, device=device)
    
    print('\n' + '=' * 50)
    print('GRU4Rec Results on Kuairec first_average')
    print('=' * 50)
    for k in [5, 10, 20]:
        print(f'Recall@{k}: {results[f"recall@{k}"]:.4f}')
        print(f'MRR@{k}:    {results[f"mrr@{k}"]:.4f}')
        print(f'NDCG@{k}:   {results[f"ndcg@{k}"]:.4f}')
    
    with open(args.output, 'w') as f:
        f.write('GRU4Rec Results on Kuairec first_average\n')
        f.write(f'Training time: {train_time:.1f}s\n\n')
        for k in [5, 10, 20]:
            f.write(f'Recall@{k}: {results[f"recall@{k}"]:.4f}\n')
            f.write(f'MRR@{k}:    {results[f"mrr@{k}"]:.4f}\n')
            f.write(f'NDCG@{k}:   {results[f"ndcg@{k}"]:.4f}\n')


if __name__ == '__main__':
    main()
