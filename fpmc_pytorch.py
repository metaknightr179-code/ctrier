#!/usr/bin/env python3
"""
FPMC PyTorch implementation for KuaiRec baseline comparison.
FPMC: Factorizing Personalized Markov Chains for Next-Basket Recommendation
(Rendle, Freudenthaler, Schmidt-Thieme, WWW 2010).

Score(u, prev, next) = <U[u], I[next]> + <P[prev], N[next]>
Trained with BPR on every consecutive item pair (negative sampling).
Extremely fast: no recurrence/attention — full-rank eval is two matmuls.
Computes the same metrics as TRIER (recall, MRR, NDCG, ILD, CS, CC).
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script import evaluate_function_with_full, get_metrics_full, get_cates_map


class FPMCModel(nn.Module):
    def __init__(self, user_num, item_num, dim=64):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        # 4 embedding tables: user, item (as next), prev-item, next-item transition
        self.user_emb = nn.Embedding(user_num, dim)
        self.item_emb = nn.Embedding(item_num + 1, dim, padding_idx=0)
        self.prev_emb = nn.Embedding(item_num + 1, dim, padding_idx=0)
        self.next_emb = nn.Embedding(item_num + 1, dim, padding_idx=0)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        for emb in (self.item_emb, self.prev_emb, self.next_emb):
            nn.init.normal_(emb.weight[1:], std=0.01)

    def score_pairs(self, users, prevs, items):
        """Score for (user, prev_item, candidate_item) triples.
        users/prevs/items: (B,) -> scores (B,)"""
        u = self.user_emb(users)
        l = self.prev_emb(prevs)
        i = self.item_emb(items)
        n = self.next_emb(items)
        return (u * i).sum(-1) + (l * n).sum(-1)

    def score_all(self, users, prevs):
        """Score every item for a batch of (user, prev_item).
        users/prevs: (B,) -> scores (B, item_num+1)"""
        u = self.user_emb(users)           # (B, D)
        l = self.prev_emb(prevs)           # (B, D)
        # <U[u], I> over all items + <P[prev], N> over all items
        scores = u @ self.item_emb.weight.T + l @ self.next_emb.weight.T  # (B, item_num+1)
        scores[:, 0] = float('-inf')  # pad
        return scores


def load_lines(data_file):
    """Read sequences; first token is user_id. Returns list of (user, [items...])."""
    out = []
    with open(data_file, 'r') as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            if len(tokens) >= 3:  # user + at least 2 items
                out.append((tokens[0], tokens[1:]))
    return out


def build_transitions(sequences, maxlen=50):
    """Every consecutive item pair is a positive transition.
    Returns tensors (users, prevs, nexts)."""
    users, prevs, nexts = [], [], []
    for u, items in sequences:
        seq = items[-(maxlen + 1):]
        for t in range(1, len(seq)):
            users.append(u)
            prevs.append(seq[t - 1])
            nexts.append(seq[t])
    return (torch.tensor(users, dtype=torch.long),
            torch.tensor(prevs, dtype=torch.long),
            torch.tensor(nexts, dtype=torch.long))


def build_eval_tensors(sequences):
    """Predict last item from preceding sequence's last item.
    Returns tensors (users, prevs, targets)."""
    users, prevs, targets = [], [], []
    for u, items in sequences:
        users.append(u)
        prevs.append(items[-2])
        targets.append(items[-1])
    return (torch.tensor(users, dtype=torch.long),
            torch.tensor(prevs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


def bpr_evaluate(model, users, prevs, targets, batch_size, device,
                 cat_map=None, cat_num=0, item2vec=None):
    """Full-rank evaluation: recall/MRR/NDCG/ILD/CS/CC."""
    model.eval()
    total_result = []
    n = users.size(0)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            u = users[s:s + batch_size].to(device)
            l = prevs[s:s + batch_size].to(device)
            t = targets[s:s + batch_size].to(device)
            scores = model.score_all(u, l)          # (B, item_num+1)
            _, rec = scores.topk(k=20, dim=-1)      # (B, 20)
            result = evaluate_function_with_full(
                t, rec, cat_map=cat_map, cat_num=cat_num, item2vec=item2vec)
            total_result.extend(result)

    keys = ['recall@5_f', 'recall@10_f', 'recall@20_f',
            'mrr@5_f', 'mrr@10_f', 'mrr@20_f',
            'ndcg@5_f', 'ndcg@10_f', 'ndcg@20_f',
            'ILD@5', 'ILD@10', 'ILD@20',
            'CS@5', 'CS@10', 'CS@20',
            'CC@5', 'CC@10', 'CC@20']
    return {k: get_metrics_full(k, total_result) for k in keys}


def train_fpmc(train_file, item_num, epochs, batch_size, lr, dim, n_neg, maxlen,
               ckpt_dir, device, valid_seqs=None, patience=100, min_delta=0.0001):
    torch.manual_seed(42)
    np.random.seed(42)

    train_seqs = load_lines(train_file)
    user_num = max(u for u, _ in train_seqs) + 1
    print(f'Loaded {len(train_seqs)} sequences, {user_num} users')

    u_tr, l_tr, i_tr = build_transitions(train_seqs, maxlen)
    print(f'{u_tr.size(0)} transitions, {n_neg} negative samples each')

    loader = DataLoader(TensorDataset(u_tr, l_tr, i_tr),
                        batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=True)

    model = FPMCModel(user_num, item_num, dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Validation tensors for early stopping
    val_tensors = None
    if valid_seqs is not None:
        vu, vl, vt = build_eval_tensors(valid_seqs)
        val_tensors = (vu, vl, vt)

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'fpmc_best.pth')
    best_recall = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        t0 = time.time()
        for u, l, pos in loader:
            u = u.to(device)
            l = l.to(device)
            pos = pos.to(device)
            B = pos.size(0)

            # n_neg negatives per positive: (B, K)
            neg = torch.randint(1, item_num + 1, (B, n_neg), device=device)

            pos_score = model.score_pairs(u, l, pos).unsqueeze(1)      # (B,1)
            neg_score = (model.user_emb(u).unsqueeze(1) * model.item_emb(neg)).sum(-1) \
                      + (model.prev_emb(l).unsqueeze(1) * model.next_emb(neg)).sum(-1)  # (B,K)
            # BPR: -log sigmoid(x_pos - x_neg)
            loss = -F.logsigmoid(pos_score - neg_score).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)

        if val_tensors is not None:
            val_metrics = bpr_evaluate(model, *val_tensors, batch_size, device)
            monitor = val_metrics['recall@10_f']
            improved = monitor > best_recall + min_delta
            print(f'Epoch {epoch}/{epochs}, BPR Loss: {avg_loss:.4f}, '
                  f'Val R@10: {monitor:.4f} (best {max(best_recall,0):.4f} ep {best_epoch}), '
                  f'{time.time()-t0:.1f}s/epoch', flush=True)
        else:
            improved = avg_loss < -best_recall  # no valid set: save every epoch
            monitor = avg_loss
            print(f'Epoch {epoch}/{epochs}, BPR Loss: {avg_loss:.4f}, '
                  f'{time.time()-t0:.1f}s/epoch', flush=True)

        if improved:
            best_recall = monitor
            best_epoch = epoch
            torch.save({'state_dict': model.state_dict(),
                        'user_num': user_num, 'item_num': item_num, 'dim': dim},
                       ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if val_tensors is not None and patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} (patience {patience})', flush=True)
                break

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser(description='FPMC baseline')
    parser.add_argument('--train_file', required=False)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--item_num', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--n_neg', type=int, default=10, help='Negatives per positive')
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--cat', type=str, default=None, help='Category file')
    parser.add_argument('--n_cat', type=int, default=0, help='Number of categories')
    parser.add_argument('--vec', type=str, default=None, help='Item2vec .npy file')
    parser.add_argument('--output', default='fpmc_results.txt')
    parser.add_argument('--ckpt_dir', default='./save_fpmc')
    parser.add_argument('--valid_file', default=None, help='Validation file for early stopping')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
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
        ckpt_path = args.ckpt_path or os.path.join(args.ckpt_dir, 'fpmc_best.pth')
        if not os.path.exists(ckpt_path):
            print(f'ERROR: Checkpoint not found: {ckpt_path}')
            sys.exit(1)
        print(f'Eval-only mode: loading checkpoint from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = FPMCModel(ckpt['user_num'], ckpt['item_num'], dim=ckpt['dim']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        train_time = 0.0
    else:
        if not args.train_file:
            parser.error('--train_file is required when not in eval_only mode')
        valid_seqs = load_lines(args.valid_file) if args.valid_file else None
        start_time = time.time()
        model = train_fpmc(args.train_file, args.item_num, args.epochs,
                           args.batch_size, args.lr, args.dim, args.n_neg, args.maxlen,
                           args.ckpt_dir, device,
                           valid_seqs=valid_seqs, patience=args.patience)
        train_time = time.time() - start_time

    # Evaluate
    test_seqs = load_lines(args.test_file)
    u_te, l_te, t_te = build_eval_tensors(test_seqs)
    results = bpr_evaluate(model, u_te, l_te, t_te, args.batch_size, device,
                           cat_map=cat_map, cat_num=args.n_cat, item2vec=item2vec)

    # Print results
    print('\n' + '=' * 50)
    print('FPMC Results')
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
        f.write(f'FPMC Results\n')
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
