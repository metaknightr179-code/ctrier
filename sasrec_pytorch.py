#!/usr/bin/env python3
"""
SASRec PyTorch implementation for baseline comparison.
Self-Attentive Sequential Recommendation.

The evaluate_sasrec function computes accuracy (Recall/MRR/NDCG) AND diversity
(ILD, CS, CC) metrics using the same shared evaluator as TRIER and GRU4Rec,
reusing the project's pre-trained category-based item2vec vectors.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Shared metric functions from the TRIER codebase — compute ILD, CS, CC from ranked
# item lists using the same category-based item2vec vectors that TRIER itself uses.
from script import evaluate_function_with_full, get_metrics_full, get_cates_map


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
            tokens = line.strip().split()
            items = list(map(int, tokens[1:]))  # skip user_id (first token)
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


class EvalDataset(Dataset):
    """SASRec evaluation dataset — predict last item from preceding sequence.

    Matches GRU4Rec's EvalDataset convention: strips user_id (first token),
    truncates to the last (maxlen-1) items, and pads on the left with 0.
    Also returns the set of 1-indexed seen item IDs so that downstream code
    can mask them from the ranking (otherwise the baseline would trivially
    recommend items already in the input sequence).
    """
    def __init__(self, data_file, maxlen=50):
        self.data = []
        self.maxlen = maxlen
        with open(data_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                items = list(map(int, tokens[1:]))  # skip user_id
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
            set(input_seq),  # 1-indexed item IDs seen in input, for masking
        )


def train_sasrec(train_file, item_num, epochs=20, batch_size=64, lr=0.001, maxlen=50,
                 device='cuda', ckpt_dir='.', valid_file=None, patience=50, min_delta=0.0001):
    print(f'Loading training data...')
    train_sequences = load_sequences(train_file)
    print(f'Loaded {len(train_sequences)} valid sequences')

    print('Creating training batches...')
    train_batches = create_batches(train_sequences, batch_size, maxlen)
    print(f'Created {len(train_batches)} batches')

    valid_batches = None
    if valid_file:
        valid_sequences = load_sequences(valid_file)
        valid_batches = create_batches(valid_sequences, batch_size, maxlen)
        print(f'Created {len(valid_batches)} validation batches')

    model = SASRecModel(item_num, hidden_units=50, num_blocks=2, num_heads=1,
                        dropout_rate=0.5, maxlen=maxlen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'sasrec_best.pth')

    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

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

        # Validation-based monitoring
        if valid_batches:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vb_input, vb_target in valid_batches:
                    vb_input = vb_input.to(device)
                    vb_target = vb_target.to(device)
                    vb_logits = model(vb_input)
                    val_loss += criterion(vb_logits, vb_target).item()
            val_loss /= len(valid_batches)
            monitor_loss = val_loss
            print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Val: {val_loss:.4f}, Best: {best_loss:.4f} (ep {best_epoch})')
        else:
            monitor_loss = avg_loss
            print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f} (ep {best_epoch})')

        if monitor_loss < best_loss - min_delta:
            best_loss = monitor_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} (patience {patience})')
                break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    return model


def evaluate_sasrec(model, test_file, item_num, maxlen=50, device='cuda',
                    batch_size=256, cat_map=None, cat_num=0, item2vec=None):
    """Evaluate SASRec with accuracy AND diversity metrics.

    Model logits are shape (B, item_num) where index 0 maps to item 1, so we
    +1 after topk to recover 1-indexed item IDs — same convention used by
    cat_map keys and item2vec rows. Seen items are masked to -inf before
    ranking so diversity is computed on the exact top-20 list the baseline
    actually recommends (not a contaminated list with already-seen items).
    """
    print(f'\nEvaluating...')
    eval_ds = EvalDataset(test_file, maxlen)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    total_result = []

    with torch.no_grad():
        for batch_idx, (batch_input, batch_target, batch_seen) in enumerate(eval_loader):
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            logits = model(batch_input)  # (B, item_num), 0-indexed

            # Mask items already in each user's input sequence (they'd trivially
            # appear at top of the ranking otherwise)
            for i, seen in enumerate(batch_seen):
                for item in seen:
                    if item > 0 and (item - 1) < logits.shape[1]:
                        logits[i, item - 1] = float('-inf')

            # topk returns 0-indexed logit positions; +1 to get 1-indexed item IDs
            _, ranked_0idx = logits.topk(k=20, dim=-1)       # (B, 20)
            ranked_items = ranked_0idx + 1                    # (B, 20), item IDs

            result = evaluate_function_with_full(
                batch_target, ranked_items,
                cat_map=cat_map, cat_num=cat_num, item2vec=item2vec,
            )
            total_result.extend(result)

            if (batch_idx + 1) % 100 == 0:
                print(f'  Evaluated {(batch_idx + 1) * batch_size}/{len(eval_ds)}...')

    # Aggregate every metric the shared evaluator produces
    metrics = {}
    for name in ['recall@5_f', 'recall@10_f', 'recall@20_f',
                 'mrr@5_f', 'mrr@10_f', 'mrr@20_f',
                 'ndcg@5_f', 'ndcg@10_f', 'ndcg@20_f',
                 'ILD@5', 'ILD@10', 'ILD@20',
                 'CS@5', 'CS@10', 'CS@20',
                 'CC@5', 'CC@10', 'CC@20']:
        metrics[name] = get_metrics_full(name, total_result)

    # Plain (no _f suffix) aliases so both TRIER-style and SASRec-style callers work
    for k in [5, 10, 20]:
        metrics[f'recall@{k}'] = metrics[f'recall@{k}_f']
        metrics[f'mrr@{k}']    = metrics[f'mrr@{k}_f']
        metrics[f'ndcg@{k}']   = metrics[f'ndcg@{k}_f']

    return metrics


def main():
    parser = argparse.ArgumentParser(description='SASRec baseline')
    parser.add_argument('--train_file', required=False)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--item_num', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--valid_file', default=None, help='Validation file for early stopping')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--output', default='sasrec_results.txt')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only evaluate saved checkpoint')
    parser.add_argument('--ckpt_path', default='sasrec_best.pth', help='Path to checkpoint file')
    parser.add_argument('--ckpt_dir', default='.', help='Directory to save checkpoint')
    # Diversity-metric plumbing (same as GRU4Rec)
    parser.add_argument('--cat', default=None, help='Category file for diversity metrics')
    parser.add_argument('--n_cat', type=int, default=0, help='Number of categories')
    parser.add_argument('--vec', default=None, help='Item2vec .npy file for ILD/CS')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Category mapping (needed for CC@K)
    cat_map = None
    if args.cat and os.path.exists(args.cat):
        cat_map = get_cates_map(args.cat)
        print(f'Loaded category mapping from {args.cat}')

    # Pre-trained category-based item2vec (needed for ILD/CS@K)
    item2vec = None
    for vec_path in [args.vec, './KuaiRec_variants/kuairec_vec.npy', './kuairec_vec.npy']:
        if vec_path and os.path.exists(vec_path):
            item2vec = torch.tensor(np.load(vec_path))
            print(f'Loaded item embeddings from {vec_path}')
            break
    if item2vec is not None and torch.cuda.is_available():
        item2vec = item2vec.to(device)

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
        model = train_sasrec(args.train_file, args.item_num,
                             epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, maxlen=args.maxlen, device=device,
                             ckpt_dir=ckpt_dir,
                             valid_file=args.valid_file, patience=args.patience)
        train_time = time.time() - start_time
        torch.save(model.state_dict(), ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}')

    # Evaluate with both accuracy AND diversity metrics
    results = evaluate_sasrec(model, args.test_file, args.item_num, maxlen=args.maxlen,
                              device=device, batch_size=args.batch_size,
                              cat_map=cat_map, cat_num=args.n_cat, item2vec=item2vec)

    # Print results (accuracy first, then diversity)
    print('\n' + '=' * 50)
    print('SASRec Results')
    print('=' * 50)
    for k in [5, 10, 20]:
        print(f'Recall@{k}: {results[f"recall@{k}_f"]:.4f}')
        print(f'MRR@{k}:    {results[f"mrr@{k}_f"]:.4f}')
        print(f'NDCG@{k}:   {results[f"ndcg@{k}_f"]:.4f}')
    for k in [5, 10, 20]:
        print(f'ILD@{k}:    {results[f"ILD@{k}"]:.4f}')
        print(f'CS@{k}:     {results[f"CS@{k}"]:.4f}')
        print(f'CC@{k}:     {results[f"CC@{k}"]:.4f}')

    # Save results in key:value format — parse_baseline_result in analyze_results.py
    # matches "Recall/MRR/NDCG/ILD/CS/CC@K: VALUE" lines
    with open(args.output, 'w') as f:
        f.write(f'SASRec Results on {args.test_file}\n')
        f.write(f'Training time: {train_time:.1f}s\n')
        if cat_map is not None:
            f.write(f'Categories: {args.n_cat} (loaded)\n')
        if item2vec is not None:
            f.write(f'Item2vec: loaded\n')
        f.write('\n')
        for k in [5, 10, 20]:
            f.write(f'Recall@{k}: {results[f"recall@{k}_f"]:.4f}\n')
            f.write(f'MRR@{k}:    {results[f"mrr@{k}_f"]:.4f}\n')
            f.write(f'NDCG@{k}:   {results[f"ndcg@{k}_f"]:.4f}\n')
        for k in [5, 10, 20]:
            f.write(f'ILD@{k}:    {results[f"ILD@{k}"]:.4f}\n')
            f.write(f'CS@{k}:     {results[f"CS@{k}"]:.4f}\n')
            f.write(f'CC@{k}:     {results[f"CC@{k}"]:.4f}\n')


if __name__ == '__main__':
    main()
