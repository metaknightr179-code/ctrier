#!/usr/bin/env python3
"""
Recalculate Consecutive Similarity (CS) of BASELINE models (SASRec, GRU4Rec)
using three different item-vector sources:

  1. multihot : binary multi-hot category vectors (KuaiRec_variants/kuairec_vec.npy)
                — this is what CS currently uses in evaluate_function_with_full
  2. typeemb  : learned RecFormer-style type embeddings extracted from a trained
                TRIER-PT checkpoint (type_embedding.weight); each item's vector =
                mean of its categories' type embeddings
  3. itememb  : the baseline model's OWN learned item embeddings
                (SASRec item_embedding / GRU4Rec embedding)

CS@k = mean cosine similarity between adjacent items in the top-k recommendation.

Run in the directory that contains the Python files, KuaiRec_variants/, and
baseline checkpoint dirs (same place as gru4rec_pytorch.py runs):

  python3 recalc_cs_baselines.py
  python3 recalc_cs_baselines.py --models sasrec --pt_ckpt ./save_pt_type_fixloss_nodiv_kuairec_highest_individual/model/duorec-500.pth
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script import get_cates_map
from sasrec_pytorch import SASRecModel
from gru4rec_pytorch import GRU4RecModel, EvalDataset
from torch.utils.data import DataLoader

VARIANTS = [
    "kuairec_highest_individual",
    "kuairec_highest_average",
    "kuairec_first_individual",
    "kuairec_first_average",
]


def load_test_lines(test_file):
    """Same loading as baselines: NO user_id stripping (baselines keep all tokens)."""
    seqs = []
    with open(test_file, "r") as f:
        for line in f:
            items = [int(x) for x in line.strip().split()]
            if len(items) >= 2:
                seqs.append(items)
    return seqs


# ---------------------------------------------------------------------------
# Recommendation-list generation (replicates baseline eval logic exactly)
# ---------------------------------------------------------------------------
def gen_sasrec_lists(model, seqs, item_num, maxlen, device, batch_size=256):
    model.eval()
    rec_lists = []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start:start + batch_size]
            inputs, masks = [], []
            for seq in batch:
                input_seq = seq[:-1][-(maxlen - 1):]
                padded = [0] * (maxlen - len(input_seq)) + input_seq
                inputs.append(padded)
                masks.append(set(x for x in input_seq if x > 0))
            input_tensor = torch.LongTensor(inputs).to(device)
            logits = model(input_tensor)  # (B, item_num) — indices 0..item_num-1 -> items 1..item_num
            for i, seen in enumerate(masks):
                row = logits[i].clone()
                for item in seen:
                    row[item - 1] = float("-inf")
                ranked = row.argsort(descending=True)[:20] + 1  # back to item IDs
                rec_lists.append(ranked.cpu().numpy())
    return rec_lists


def gen_gru4rec_lists(model, test_file, maxlen, device, batch_size=256):
    model.eval()
    dataset = EvalDataset(test_file, maxlen)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rec_lists = []
    with torch.no_grad():
        for batch_input, _ in dataloader:
            batch_input = batch_input.to(device)
            logits = model(batch_input)  # (B, L, item_num+1)
            last_logits = logits[:, -1, :]
            last_logits[:, 0] = float("-inf")
            _, rec = last_logits.topk(k=20, dim=-1)
            rec_lists.extend(rec.cpu().numpy())
    return rec_lists


# ---------------------------------------------------------------------------
# Item-vector sources
# ---------------------------------------------------------------------------
def load_multihot_vecs(path, item_num):
    vec = torch.tensor(np.load(path), dtype=torch.float32)
    if vec.shape[0] < item_num + 1:
        pad = torch.zeros(item_num + 1 - vec.shape[0], vec.shape[1])
        vec = torch.cat([vec, pad], dim=0)
    return vec


def build_typeemb_vecs(pt_ckpt_path, cat_map, item_num, n_cat, hidden=64):
    """Per-item dense type vector from a TRIER-PT checkpoint:
    vec[i] = mean over categories c of item i of type_embedding.weight[c+1]."""
    ckpt = torch.load(pt_ckpt_path, map_location="cpu")
    if "type_embedding.weight" not in ckpt:
        raise KeyError(f"{pt_ckpt_path} has no type_embedding.weight (is it a no-type checkpoint?)")
    type_w = ckpt["type_embedding.weight"].float()  # [n_cat+1, hidden]
    vecs = torch.zeros(item_num + 1, type_w.shape[1])
    for item_id, cats in cat_map.items():
        if 0 < item_id <= item_num and len(cats) > 0:
            idx = torch.tensor([c + 1 for c in cats if 0 <= c < n_cat], dtype=torch.long)
            if len(idx) > 0:
                vecs[item_id] = type_w[idx].mean(dim=0)
    return vecs


def model_item_vecs(model, model_name):
    if model_name == "sasrec":
        return model.item_embedding.weight.detach().float().cpu()
    return model.embedding.weight.detach().float().cpu()


def find_pt_checkpoint(variant):
    """Auto-pick a type-enabled PT checkpoint (latest epoch, preferred dirs first)."""
    patterns = [
        f"./save_pt_fixrt_lamb0005_{variant}/model/duorec-*.pth",
        f"./save_pt_type_fixloss_lamb0005_{variant}/model/duorec-*.pth",
        f"./save_pt_fixrt_nodiv_{variant}/model/duorec-*.pth",
        f"./save_pt_type_fixloss_nodiv_{variant}/model/duorec-*.pth",
        "./save_pt_fixrt_lamb0005_*/model/duorec-*.pth",
        "./save_pt_type_*/model/duorec-*.pth",
        "./save_pt_*/model/duorec-*.pth",
    ]
    best = None
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if files:
            best = max(files, key=lambda p: int(os.path.basename(p).split("-")[-1].split(".")[0]))
            break
    return best


# ---------------------------------------------------------------------------
# CS computation
# ---------------------------------------------------------------------------
def cs_from_vecs(rec_lists, vecs, ks=(5, 10, 20)):
    """Mean cosine similarity between adjacent items in each top-k list."""
    sums = {k: 0.0 for k in ks}
    counts = {k: 0 for k in ks}
    for rec in rec_lists:
        for k in ks:
            ids = torch.tensor(rec[:k], dtype=torch.long)
            v = vecs[ids]  # [k, d]
            v1, v2 = v[:-1], v[1:]
            # skip pairs where either vector is zero (undefined cosine)
            norm = v1.norm(dim=1) * v2.norm(dim=1)
            valid = norm > 1e-8
            if valid.sum() > 0:
                sim = F.cosine_similarity(v1[valid], v2[valid], dim=-1)
                sums[k] += sim.mean().item()
                counts[k] += 1
    return {k: (sums[k] / counts[k] if counts[k] else 0.0) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--models", nargs="*", default=["sasrec", "gru4rec"])
    ap.add_argument("--item_num", type=int, default=10728)
    ap.add_argument("--n_cat", type=int, default=31)
    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--pt_ckpt", default="auto",
                    help="TRIER-PT checkpoint for type embeddings; 'auto' = search save_pt_* dirs")
    ap.add_argument("--output", default="cs_recalc_results.txt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rows = []
    for variant in args.variants:
        data_dir = f"./KuaiRec_variants/{variant}"
        test_file = f"{data_dir}/test-v0.txt"
        cat_file = f"{data_dir}/kuairec_cate.txt"
        vec_file = "./KuaiRec_variants/kuairec_vec.npy"

        if not os.path.exists(test_file):
            print(f"[skip] {variant}: no test file {test_file}")
            continue

        cat_map = get_cates_map(cat_file) if os.path.exists(cat_file) else {}

        # --- shared vector sources ---
        multihot = load_multihot_vecs(vec_file, args.item_num) if os.path.exists(vec_file) else None
        if multihot is None:
            print(f"[warn] {vec_file} not found — multihot CS unavailable")

        pt_ckpt = args.pt_ckpt
        if pt_ckpt == "auto":
            pt_ckpt = find_pt_checkpoint(variant)
        typeemb = None
        if pt_ckpt and os.path.exists(pt_ckpt):
            try:
                typeemb = build_typeemb_vecs(pt_ckpt, cat_map, args.item_num, args.n_cat)
                print(f"[{variant}] type embeddings from {pt_ckpt}")
            except Exception as e:
                print(f"[{variant}] type embeddings unavailable: {e}")
        else:
            print(f"[{variant}] no PT checkpoint found for type embeddings")

        for model_name in args.models:
            # --- load model + generate rec lists ---
            if model_name == "sasrec":
                ckpt = f"./save_sasrec_{variant}/sasrec_best.pth"
                if not os.path.exists(ckpt):
                    ckpt = "./sasrec_best.pth"
                if not os.path.exists(ckpt):
                    print(f"[skip] sasrec/{variant}: no checkpoint")
                    continue
                model = SASRecModel(args.item_num, hidden_units=50, num_blocks=2,
                                    num_heads=1, dropout_rate=0.5, maxlen=args.maxlen).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                seqs = load_test_lines(test_file)
                rec_lists = gen_sasrec_lists(model, seqs, args.item_num, args.maxlen, device)
            elif model_name == "gru4rec":
                ckpt = f"./save_gru4rec_{variant}/gru4rec_best.pth"
                if not os.path.exists(ckpt):
                    print(f"[skip] gru4rec/{variant}: no checkpoint")
                    continue
                model = GRU4RecModel(args.item_num, embedding_dim=64, hidden_dim=64).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                rec_lists = gen_gru4rec_lists(model, test_file, args.maxlen, device)
            else:
                print(f"[skip] unknown model {model_name}")
                continue

            itememb = model_item_vecs(model, model_name)

            print(f"\n=== {model_name} / {variant} ({len(rec_lists)} users) ===")
            for source, vecs in [("multihot", multihot), ("typeemb", typeemb), ("itememb", itememb)]:
                if vecs is None:
                    continue
                cs = cs_from_vecs(rec_lists, vecs)
                print(f"  CS ({source:8s}):  @5={cs[5]:.4f}  @10={cs[10]:.4f}  @20={cs[20]:.4f}")
                rows.append((model_name, variant, source, cs[5], cs[10], cs[20]))
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        f.write("model\tvariant\tsource\tCS@5\tCS@10\tCS@20\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\n")
    print(f"\nSaved table to {args.output}")


if __name__ == "__main__":
    main()
