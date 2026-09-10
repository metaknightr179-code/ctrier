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

    def _encode(self, input_seq):
        """Run embeddings + transformer encoder. Returns (B, L, H)."""
        B, L = input_seq.shape
        pos_indices = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, -1)
        seq_emb = self.item_embedding(input_seq) + self.position_embedding(pos_indices)
        seq_emb = self.ln(seq_emb)
        seq_emb = self.dropout(seq_emb)
        # Padding mask: True = positions to ignore (pad token 0)
        padding_mask = (input_seq == 0)  # (B, L)
        encoded = self.encoder(seq_emb, src_key_padding_mask=padding_mask)
        return encoded

    def forward(self, input_seq, pos_indices=None):
        """Forward pass (eval): full logits at every position.
        Returns logits: (B, L, item_num+1)
        """
        encoded = self._encode(input_seq)  # (B, L, H)
        logits = self.out_proj(encoded)    # (B, L, item_num+1)
        return logits

    def forward_mlm(self, masked_seq, mask_mask):
        """Forward pass (training): compute logits ONLY at masked positions.
        Avoids materializing the huge (B, L, V) output tensor.
          masked_seq: (B, L) input with mask_token substituted
          mask_mask:  (B, L) bool — True where masked
        Returns logits: (M, item_num+1) over the M masked positions only.
        """
        encoded = self._encode(masked_seq)   # (B, L, H)
        hidden = encoded[mask_mask]          # (M, H) gather masked positions
        logits = self.out_proj(hidden)       # (M, item_num+1)
        return logits

    def mask_sequence(self, input_seq):
        """Fully vectorized MLM masking (one GPU sync for the whole batch).
        Returns: (masked_seq, mask_mask, targets).
          masked_seq: (B, L) with mask_token substituted
          mask_mask:  (B, L) bool — True where masked
          targets:    (B, L) original item IDs (0 where not masked)
        """
        B, L = input_seq.shape
        device = input_seq.device

        valid = input_seq > 0                                  # (B, L)
        n_valid = valid.sum(dim=1)                             # (B,)
        n_masks = torch.clamp((n_valid.float() * self.mask_prob).round().long(),
                              min=1)                           # (B,)
        max_masks = int(n_masks.max().item())                  # single sync

        # Random score per position; padding ranks last
        rand = torch.rand(B, L, device=device).masked_fill(~valid, -1e9)
        _, top_idx = rand.topk(max_masks, dim=1)              # (B, max_masks)

        # Keep only the first n_masks[b] entries per row
        keep = torch.arange(max_masks, device=device).unsqueeze(0) < n_masks.unsqueeze(1)
        rows = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_masks)
        flat_idx = rows[keep] * L + top_idx[keep]            # (M,) flat indices

        mask_mask = torch.zeros(B * L, dtype=torch.bool, device=device)
        mask_mask[flat_idx] = True
        mask_mask = mask_mask.view(B, L)

        targets = input_seq.clone()
        targets[~mask_mask] = 0

        masked_seq = input_seq.clone()
        masked_seq[mask_mask] = self.mask_token

        return masked_seq, mask_mask, targets


def load_sequences(data_file):
    sequences = []
    with open(data_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 3:
                continue
            items = [int(x) for x in tokens[1:]]  # strip first token (user_id)
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
                   lr=0.001, maxlen=50, device='cuda', ckpt_dir='.',
                   valid_file=None, patience=0):
    set_seed(42)
    print(f'Loading training data...')
    train_sequences = load_sequences(train_file)
    print(f'Loaded {len(train_sequences)} valid sequences')

    inputs, _ = create_train_data(train_sequences, maxlen=maxlen)
    print(f'Created {len(inputs)} training examples')

    train_batches = create_batches(inputs, batch_size)
    print(f'{len(train_batches)} batches of size {batch_size}')

    # Validation batches for early stopping
    valid_batches = []
    if valid_file and patience > 0:
        valid_sequences = load_sequences(valid_file)
        valid_inputs, _ = create_train_data(valid_sequences, maxlen=maxlen)
        valid_batches = create_batches(valid_inputs, batch_size)
        print(f'Validation: {len(valid_batches)} batches')

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
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_masked = 0
        np.random.shuffle(train_batches)

        t0 = time.time()
        for batch in train_batches:
            batch = batch.to(device)

            # Vectorized masking (single GPU sync for the whole batch)
            masked_batch, mask_mask, targets = model.mask_sequence(batch)

            # Logits only at masked positions: (M, V) instead of (B*L, V)
            masked_logits = model.forward_mlm(masked_batch, mask_mask)  # (M, V)
            masked_targets = targets[mask_mask]                        # (M,)
            loss = criterion(masked_logits, masked_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n_masked = int(masked_targets.size(0))
            total_loss += loss.item() * n_masked
            num_masked += n_masked

        avg_loss = total_loss / max(num_masked, 1)

        # Validation for early stopping
        eval_loss = avg_loss
        if valid_batches:
            model.eval()
            v_loss, v_masked = 0.0, 0
            with torch.no_grad():
                for batch in valid_batches:
                    batch = batch.to(device)
                    masked_batch, mask_mask, targets = model.mask_sequence(batch)
                    masked_logits = model.forward_mlm(masked_batch, mask_mask)
                    masked_targets = targets[mask_mask]
                    loss = criterion(masked_logits, masked_targets)
                    n = int(masked_targets.size(0))
                    v_loss += loss.item() * n
                    v_masked += n
            eval_loss = v_loss / max(v_masked, 1)

        improved = eval_loss < best_loss
        if improved:
            best_loss = eval_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1

        val_tag = f', Val Loss: {eval_loss:.4f}' if valid_batches else ''
        print(f'Epoch {epoch}/{epochs}, MLM Loss: {avg_loss:.4f}{val_tag}, '
              f'Best: {best_loss:.4f} (ep {best_epoch}), {time.time()-t0:.1f}s/epoch',
              flush=True)

        if patience > 0 and patience_counter >= patience:
            print(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
            break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


class BertEvalDataset(torch.utils.data.Dataset):
    """Batched eval: left-padded sequence + [MASK] at last position."""
    def __init__(self, data_file, maxlen=50):
        self.rows = []
        self.maxlen = maxlen
        with open(data_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                items = list(map(int, tokens[1:]))  # skip user_id
                if len(items) >= 2:
                    self.rows.append(items)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq = self.rows[idx]
        input_seq = seq[:-1][-(self.maxlen - 1):]
        target = seq[-1]
        return input_seq, target


def _collate_eval(batch, mask_token, maxlen):
    inputs, targets = [], []
    for input_seq, target in batch:
        seq = input_seq + [mask_token]
        padded = [0] * (maxlen - len(seq)) + seq
        inputs.append(padded)
        targets.append(target)
    return (torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


def evaluate_bert4rec(model, test_file, item_num, maxlen=50, batch_size=256,
                      device='cuda', cat_map=None, cat_num=0, item2vec=None):
    """Batched full-rank eval: recall/MRR/NDCG + ILD/CS/CC via shared evaluator."""
    from functools import partial
    from script import evaluate_function_with_full, get_metrics_full
    from torch.utils.data import DataLoader

    print(f'\nEvaluating BERT4Rec...')
    dataset = BertEvalDataset(test_file, maxlen)
    collate = partial(_collate_eval, mask_token=model.mask_token, maxlen=maxlen)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)

    model.eval()
    total_result = []
    with torch.no_grad():
        for input_tensor, targets in dataloader:
            input_tensor = input_tensor.to(device)
            targets = targets.to(device)
            logits = model(input_tensor)           # (B, L, item_num+1)
            last_logits = logits[:, -1, :]         # MASK is always last position
            last_logits[:, 0] = float('-inf')      # exclude padding
            _, rec_list = last_logits.topk(k=20, dim=-1)
            result = evaluate_function_with_full(
                targets, rec_list,
                cat_map=cat_map, cat_num=cat_num, item2vec=item2vec)
            total_result.extend(result)

    keys = ['recall@5_f', 'recall@10_f', 'recall@20_f',
            'mrr@5_f', 'mrr@10_f', 'mrr@20_f',
            'ndcg@5_f', 'ndcg@10_f', 'ndcg@20_f',
            'ILD@5', 'ILD@10', 'ILD@20',
            'CS@5', 'CS@10', 'CS@20',
            'CC@5', 'CC@10', 'CC@20']
    return {k: get_metrics_full(k, total_result) for k in keys}


def main():
    parser = argparse.ArgumentParser(description='BERT4Rec baseline (standalone)')
    parser.add_argument('--train_file', required=False)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--item_num', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--cat', type=str, default=None, help='Category file')
    parser.add_argument('--n_cat', type=int, default=0, help='Number of categories')
    parser.add_argument('--vec', type=str, default=None, help='Item2vec .npy file')
    parser.add_argument('--output', default='bert4rec_results.txt')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only evaluate saved checkpoint')
    parser.add_argument('--ckpt_path', default='bert4rec_best.pth', help='Path to checkpoint file')
    parser.add_argument('--ckpt_dir', default='.', help='Directory to save checkpoint')
    parser.add_argument('--valid_file', default=None, help='Validation file for early stopping')
    parser.add_argument('--patience', type=int, default=0, help='Early stop after N epochs without improvement (0=off)')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from script import get_cates_map

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    cat_map = None
    if args.cat and os.path.exists(args.cat):
        cat_map = get_cates_map(args.cat)
        print(f'Loaded category mapping from {args.cat}')

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
            ckpt_dir=ckpt_dir,
            valid_file=args.valid_file, patience=args.patience
        )
        train_time = time.time() - start_time
        torch.save(model.state_dict(), ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}')

    results = evaluate_bert4rec(model, args.test_file, args.item_num,
                                maxlen=args.maxlen, batch_size=args.batch_size,
                                device=device, cat_map=cat_map, cat_num=args.n_cat,
                                item2vec=item2vec)

    print('\n' + '=' * 50)
    print('BERT4Rec Results')
    print('=' * 50)
    for k in [5, 10, 20]:
        print(f'Recall@{k}: {results[f"recall@{k}_f"]:.4f}')
        print(f'MRR@{k}:    {results[f"mrr@{k}_f"]:.4f}')
        print(f'NDCG@{k}:   {results[f"ndcg@{k}_f"]:.4f}')
    for k in [5, 10, 20]:
        print(f'ILD@{k}:    {results[f"ILD@{k}"]:.4f}')
        print(f'CS@{k}:     {results[f"CS@{k}"]:.4f}')
        print(f'CC@{k}:     {results[f"CC@{k}"]:.4f}')

    with open(args.output, 'w') as f:
        f.write(f'BERT4Rec Results\n')
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


if __name__ == '__main__':
    main()
