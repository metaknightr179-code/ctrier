#!/usr/bin/env python3
"""
BERT4Rec PyTorch implementation for Kuairec baseline comparison.
BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer.
"""
import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BERT4RecModel(nn.Module):
    def __init__(self, item_num, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.2, maxlen=50, mask_prob=0.2):
        super().__init__()
        self.item_num = item_num
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.mask_prob = mask_prob

        # Special tokens: 0 = pad, item_num+1 = mask
        self.mask_token = item_num + 1
        num_items_with_special = item_num + 2  # +pad(0) +mask

        self.item_embedding = nn.Embedding(num_items_with_special, hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(maxlen, hidden_units)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(hidden_units)
        self.out_proj = nn.Linear(hidden_units, item_num + 1)  # + pad

    def forward(self, input_seq, pos_indices=None):
        """Forward pass.
        input_seq: (B, L) with token IDs (0=pad, 1..item_num=items, item_num+1=mask)
        Returns logits: (B, L, item_num+1)
        """
        B, L = input_seq.shape

        if pos_indices is None:
            pos_indices = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, -1)

        seq_emb = self.item_embedding(input_seq) + self.position_embedding(pos_indices)
        seq_emb = self.ln(seq_emb)
        seq_emb = self.dropout(seq_emb)

        encoded = self.encoder(seq_emb)  # (B, L, H)

        logits = self.out_proj(encoded)  # (B, L, item_num+1)
        return logits

    def mask_sequence(self, input_seq):
        """Randomly mask tokens for MLM training.
        Returns: (masked_seq, mask_positions, mask_targets)
        Each of shape (B, L), (B, num_masks), (B, num_masks)
        """
        B, L = input_seq.shape
        masked_seq = input_seq.clone()
        mask_positions = []
        mask_targets = []

        for b in range(B):
            valid_mask = input_seq[b] > 0  # non-pad
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                mask_positions.append(torch.tensor([], dtype=torch.long))
                mask_targets.append(torch.tensor([], dtype=torch.long))
                continue

            num_masks = max(1, int(len(valid_indices) * self.mask_prob))
            selected = torch.randperm(len(valid_indices))[:num_masks]
            pos = valid_indices[selected]
            targets = input_seq[b, pos].clone()
            masked_seq[b, pos] = self.mask_token

            mask_positions.append(pos.cpu())
            mask_targets.append(targets.cpu())

        return masked_seq, mask_positions, mask_targets


def load_sequences(data_file):
    sequences = []
    with open(data_file, 'r') as f:
        for line in f:
            items = [int(x) for x in line.strip().split()]
            if len(items) >= 2:
                sequences.append(items)
    return sequences


def create_train_data(sequences, maxlen=50):
    """Create training examples from user sequences.
    Each sequence (i1, i2, ..., iN) becomes:
      input: [i1, i2, ..., i_{N-1}] (masked randomly)
      target: [iN] (last item for eval reference, but loss is on masked positions)
    """
    all_inputs = []
    all_target_last = []

    for seq in sequences:
        seq = seq[-maxlen:]
        if len(seq) >= 2:
            input_items = seq[:]  # Use full seq, mask random items during forward
            target_last = seq[-1]
            padded = [0] * (maxlen - len(input_items)) + input_items
            all_inputs.append(padded)
            all_target_last.append(target_last)

    return all_inputs, all_target_last


def create_batches(inputs, batch_size):
    batches = []
    for i in range(0, len(inputs), batch_size):
        batch = torch.LongTensor(inputs[i:i+batch_size])
        batches.append(batch)
    return batches


def train_bert4rec(train_file, test_file, item_num, epochs=500, batch_size=64,
                   lr=0.001, maxlen=50, device='cuda', ckpt_dir='.'):
    set_seed(42)
    print(f'Loading training data...')
    train_sequences = load_sequences(train_file)
    print(f'Loaded {len(train_sequences)} valid sequences')

    inputs, _ = create_train_data(train_sequences, maxlen=maxlen)
    print(f'Created {len(inputs)} training examples')

    train_batches = create_batches(inputs, batch_size)
    print(f'{len(train_batches)} batches of size {batch_size}')

    model = BERT4RecModel(
        item_num=item_num,
        hidden_units=64,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        maxlen=maxlen
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'bert4rec_best.pth')

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_masked = 0
        np.random.shuffle(train_batches)

        for batch in train_batches:
            batch = batch.to(device)

            # Mask random tokens
            masked_batch, mask_pos_list, mask_tgt_list = model.mask_sequence(batch)

            logits = model(masked_batch)  # (B, L, item_num+1)

            # Compute loss only on masked positions
            loss = torch.tensor(0.0, device=device)
            total_preds = 0
            for b in range(batch.size(0)):
                pos = mask_pos_list[b]
                tgt = mask_tgt_list[b]
                if len(pos) > 0:
                    pos = pos.to(device)
                    tgt = tgt.to(device)
                    pred_logits = logits[b, pos, :]  # (M, item_num+1)
                    loss = loss + criterion(pred_logits, tgt)
                    total_preds += len(pos)

            if total_preds > 0:
                loss = loss / total_preds

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * total_preds
            num_masked += total_preds

        avg_loss = total_loss / max(num_masked, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        print(f'Epoch {epoch}/{epochs}, MLM Loss: {avg_loss:.4f}, '
              f'Best: {best_loss:.4f} (ep {best_epoch})')

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate_bert4rec(model, test_file, item_num, maxlen=50, device='cuda'):
    """Evaluate BERT4Rec using [MASK] on the last position."""
    print(f'\nEvaluating BERT4Rec...')
    test_sequences = load_sequences(test_file)

    model.eval()
    all_recall = {5: [], 10: [], 20: []}
    all_mrr = {5: [], 10: [], 20: []}
    all_ndcg = {5: [], 10: [], 20: []}

    with torch.no_grad():
        for seq_idx, seq in enumerate(test_sequences):
            if len(seq) < 2:
                continue

            input_seq = seq[:-1][-(maxlen - 1):]  # all but last, clipped
            target = seq[-1]

            # Append MASK token at the end
            full_seq = input_seq + [model.mask_token]
            padded = [0] * (maxlen - len(full_seq)) + full_seq

            input_tensor = torch.LongTensor([padded]).to(device)
            logits = model(input_tensor)  # (1, L, item_num+1)

            # Predictions at the MASK position (last non-pad token)
            mask_pos = len(padded) - 1
            while mask_pos >= 0 and padded[mask_pos] == 0:
                mask_pos -= 1

            last_logits = logits[0, mask_pos, 1:]  # skip pad(0), shape: (item_num,)

            # Mask items already in input sequence
            for item in set(input_seq):
                if 1 <= item <= item_num:
                    last_logits[item - 1] = float('-inf')

            ranked = last_logits.argsort(descending=True).cpu().numpy()
            ranked_items = ranked + 1  # back to 1-indexed item IDs

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
    parser = argparse.ArgumentParser(description='BERT4Rec baseline')
    parser.add_argument('--train_file', required=False)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--item_num', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--output', default='bert4rec_results.txt')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only evaluate saved checkpoint')
    parser.add_argument('--ckpt_path', default='bert4rec_best.pth', help='Path to checkpoint file')
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
        model = BERT4RecModel(
            item_num=args.item_num,
            hidden_units=64,
            num_blocks=2,
            num_heads=2,
            dropout_rate=0.2,
            maxlen=args.maxlen
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        train_time = 0.0
    else:
        if not args.train_file:
            parser.error('--train_file is required when not in eval_only mode')
        start_time = time.time()
        model = train_bert4rec(
            args.train_file, args.test_file, args.item_num,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, maxlen=args.maxlen, device=device,
            ckpt_dir=ckpt_dir
        )
        train_time = time.time() - start_time
        # Save checkpoint to ckpt_dir
        torch.save(model.state_dict(), ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}')

    results = evaluate_bert4rec(model, args.test_file, args.item_num,
                                maxlen=args.maxlen, device=device)

    print('\n' + '=' * 50)
    print('BERT4Rec Results')
    print('=' * 50)
    for k in [5, 10, 20]:
        print(f'Recall@{k}: {results[f"recall@{k}"]:.4f}')
        print(f'MRR@{k}:    {results[f"mrr@{k}"]:.4f}')
        print(f'NDCG@{k}:   {results[f"ndcg@{k}"]:.4f}')

    with open(args.output, 'w') as f:
        f.write(f'BERT4Rec Results\n')
        f.write(f'Training time: {train_time:.1f}s\n')
        f.write(f'Epochs: {args.epochs}\n\n')
        for k in [5, 10, 20]:
            f.write(f'Recall@{k}: {results[f"recall@{k}"]:.4f}\n')
            f.write(f'MRR@{k}:    {results[f"mrr@{k}"]:.4f}\n')
            f.write(f'NDCG@{k}:   {results[f"ndcg@{k}"]:.4f}\n')


if __name__ == '__main__':
    main()
