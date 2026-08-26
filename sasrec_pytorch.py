#!/usr/bin/env python3
"""
SASRec PyTorch implementation for Kuairec baseline comparison.
Self-Attentive Sequential Recommendation.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn


class SASRecModel(nn.Module):
    def __init__(self, item_num, hidden_units=50, num_blocks=2, num_heads=1, dropout_rate=0.5, maxlen=50):
        super().__init__()
        self.item_num = item_num
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        
        self.item_embedding = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(maxlen, hidden_units)
        
        self.attention_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_units,
                    nhead=num_heads,
                    dim_feedforward=hidden_units * 2,
                    dropout=dropout_rate,
                    batch_first=True
                )
            )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(hidden_units)
        
    def forward(self, input_seq):
        seq_len = input_seq.shape[1]
        
        pos_indices = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(input_seq.shape[0], -1)
        seq_emb = self.item_embedding(input_seq) + self.position_embedding(pos_indices)
        seq_emb = self.ln(seq_emb)
        seq_emb = self.dropout(seq_emb)
        
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_seq.device), diagonal=1)
        
        for attention_layer in self.attention_layers:
            seq_emb = attention_layer(seq_emb, mask)
        
        logits = torch.matmul(seq_emb[:, -1, :], self.item_embedding.weight[1:].T)
        return logits


def load_sequences(data_file):
    sequences = []
    with open(data_file, 'r') as f:
        for line in f:
            items = [int(x) for x in line.strip().split()]
            if len(items) >= 2:
                sequences.append(items)
    return sequences


def create_batches(sequences, batch_size, maxlen=50):
    inputs = []
    targets = []
    
    for seq in sequences:
        seq = seq[-maxlen:]
        if len(seq) >= 2:
            input_seq = seq[:-1]
            target = seq[-1]
            padded = [0] * (maxlen - len(input_seq)) + input_seq
            inputs.append(padded)
            targets.append(target)
    
    batches = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = torch.LongTensor(inputs[i:i+batch_size])
        batch_targets = torch.LongTensor([t - 1 for t in targets[i:i+batch_size]])  # -1 for 0-indexed
        batches.append((batch_inputs, batch_targets))
    
    return batches


def train_sasrec(train_file, test_file, item_num, epochs=20, batch_size=64, lr=0.001, maxlen=50, device='cuda', ckpt_dir='.'):
    print(f'Loading training data...')
    train_sequences = load_sequences(train_file)
    print(f'Loaded {len(train_sequences)} valid sequences')
    
    print('Creating training batches...')
    train_batches = create_batches(train_sequences, batch_size, maxlen)
    print(f'Created {len(train_batches)} batches')
    
    model = SASRecModel(item_num, hidden_units=50, num_blocks=2, num_heads=1, 
                        dropout_rate=0.5, maxlen=maxlen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'sasrec_best.pth')
    
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        np.random.shuffle(train_batches)
        
        for batch_input, batch_target in train_batches:
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
            torch.save(model.state_dict(), ckpt_path)
        
        print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f} (ep {best_epoch})')
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    return model


def evaluate_sasrec(model, test_file, item_num, maxlen=50, device='cuda'):
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
            
            input_seq = seq[:-1][-(maxlen-1):]
            target = seq[-1]
            
            padded = [0] * (maxlen - len(input_seq)) + input_seq
            input_tensor = torch.LongTensor([padded]).to(device)
            
            logits = model(input_tensor)[0]
            
            # Mask out items in input sequence
            logits_masked = logits.clone()
            for item in set(input_seq):
                if item > 0:
                    logits_masked[item - 1] = float('-inf')
            
            ranked = logits_masked.argsort(descending=True).cpu().numpy()
            ranked_items = ranked + 1  # Convert back to item IDs
            
            for k in [5, 10, 20]:
                hit = target in ranked_items[:k]
                all_recall[k].append(1.0 if hit else 0.0)
                
                if hit:
                    rank = np.where(ranked_items[:k] == target)[0][0] + 1
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
    parser.add_argument('--train_file', required=False)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--item_num', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--output', default='sasrec_results.txt')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only evaluate saved checkpoint')
    parser.add_argument('--ckpt_path', default='sasrec_best.pth', help='Path to checkpoint file')
    parser.add_argument('--ckpt_dir', default='.', help='Directory to save checkpoint')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else '.'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_path)
    
    if args.eval_only:
        if not os.path.exists(ckpt_path):
            print(f'ERROR: Checkpoint not found: {ckpt_path}')
            sys.exit(1)
        print(f'Eval-only mode: loading checkpoint from {ckpt_path}')
        model = SASRecModel(args.item_num, hidden_units=50, num_blocks=2, num_heads=1, 
                            dropout_rate=0.5, maxlen=args.maxlen).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        train_time = 0.0
    else:
        if not args.train_file:
            parser.error('--train_file is required when not in eval_only mode')
        start_time = time.time()
        model = train_sasrec(args.train_file, args.test_file, args.item_num,
                            epochs=args.epochs, batch_size=args.batch_size, 
                            lr=args.lr, maxlen=args.maxlen, device=device,
                            ckpt_dir=ckpt_dir)
        train_time = time.time() - start_time
        # Save checkpoint to ckpt_dir
        torch.save(model.state_dict(), ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}')
    
    results = evaluate_sasrec(model, args.test_file, args.item_num, maxlen=args.maxlen, device=device)
    
    print('\n' + '=' * 50)
    print(f'SASRec Results on {args.test_file}')
    print('=' * 50)
    for k in [5, 10, 20]:
        print(f'Recall@{k}: {results[f"recall@{k}"]:.4f}')
        print(f'MRR@{k}:    {results[f"mrr@{k}"]:.4f}')
        print(f'NDCG@{k}:   {results[f"ndcg@{k}"]:.4f}')
    
    with open(args.output, 'w') as f:
        f.write(f'SASRec Results on {args.test_file}\n')
        f.write(f'Training time: {train_time:.1f}s\n\n')
        for k in [5, 10, 20]:
            f.write(f'Recall@{k}: {results[f"recall@{k}"]:.4f}\n')
            f.write(f'MRR@{k}:    {results[f"mrr@{k}"]:.4f}\n')
            f.write(f'NDCG@{k}:   {results[f"ndcg@{k}"]:.4f}\n')


if __name__ == '__main__':
    main()
