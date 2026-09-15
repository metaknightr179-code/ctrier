#!/usr/bin/env python3
"""Measure ACTUAL parameter counts per six-cell checkpoint.

Simplified from measure_model_complexity.py: only does param counting
(sum(p.numel() for p in model.parameters())) which is authoritative. No
forward-pass (those need valid side-info id ranges; param counts don't).

Usage:  cd ~/ctrier && python3 measure_model_complexity.py
Output:  prints one row per checkpoint; no files changed.
"""
import os, glob, sys, torch
sys.path.insert(0, ".")
from script import get_args
import main_pt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

VAR = "kuairec_first_average"
DATA_DIR = f"./KuaiRec_variants/{VAR}"
VEC = "./KuaiRec_variants/kuairec_vec.npy"
CATE_FILE = f"{DATA_DIR}/kuairec_cate.txt"
NEG_BIG = f"{DATA_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
N = 10728; NCAT = 31
RT_DIR = f"save_rt_fix_{VAR}"

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_latest_epoch(pt_dir):
    files = sorted(glob.glob(f"{pt_dir}/model/duorec-*.pth"),
                   key=lambda p: int(p.split("duorec-")[1].split(".pth")[0]))
    return files[-1] if files else None

def build_model(pt_dir, latest_ckpt, type_flag):
    args = get_args()
    args.device = torch.device("cpu")   # count params on CPU; no CUDA needed
    args.tf = f"{DATA_DIR}/train-v0.txt"
    args.vf = f"{DATA_DIR}/valid-v0.txt"
    args.ef = f"{DATA_DIR}/test-v0.txt"
    args.vn = NEG_BIG; args.en = NEG_BIG
    args.cat = CATE_FILE; args.vec = VEC
    args.n = N; args.n_cat = NCAT
    args.bw = 3; args.b = 256
    args.lamb = 0.01
    args.div = True
    args.t_mode = "greedy"
    args.no_type = bool(type_flag)
    args.dense = True

    model = main_pt.TRIER_PT(N, args.ln, args.hn, args.hd, args.dr, args.b, args)

    # RT model
    rt_model = None
    rt_ckpts = sorted(glob.glob(f"{RT_DIR}/model/duorec-*.pth"),
                       key=lambda p: int(p.split("duorec-")[1].split(".pth")[0]))
    if rt_ckpts:
        rt_model = main_pt.TRIER_RT(N, 2, args.hn, args.hd, args.dr, args.b, args)

    # Load PT checkpoint (optional; counting params doesn't need weights,
    # but loading confirms the checkpoint matches the model architecture)
    from script import load_state_dict_compat
    load_state_dict_compat(model, latest_ckpt, args.device)

    return model, rt_model, args

CELLS = [
    ("TRIER",       "save_pt_notype_dense_lamb001_order0",    True),
    ("TRIER-C",     "save_pt_dense_lamb001_order0",            False),
    ("TRIER-L",     "save_pt_notype_dense_lamb001_softo001",  True),
    ("TRIER-S",     "save_pt_notype_dense_lamb001_order0",    True),
    ("PACER-LS",    "save_pt_notype_dense_lamb001_softo001",  True),
    ("PACER-Full",  "save_pt_dense_lamb001_softo001",         False),
]

def breakdown(model):
    """Per-table param breakdown — useful for the paper's 'Parameter count' paragraph."""
    out = {}
    for name, mod in model.named_children():
        params = sum(p.numel() for p in mod.parameters())
        if params > 0:
            out[name] = params
    # Top-level embedding tables specifically
    extra = {}
    for etype in ["item_embedding", "type_embedding", "author_embedding", "music_embedding", "dur_embedding", "position_embedding"]:
        if hasattr(model, etype):
            emb = getattr(model, etype)
            p = sum(p.numel() for p in emb.parameters())
            extra[etype] = p
    out.update(extra)
    return out

print()
print("="*90)
print(f"{'cell':<12} {'PT ckpt':<40} {'PT total':>12} {'RT total':>12} {'PT+RT':>12} {'model arch':<30}")
print("="*90)

for cell_label, pt_suffix, is_notype in CELLS:
    pt_dir = f"{pt_suffix}_{VAR}"
    if not os.path.isdir(f"{pt_dir}/model"):
        print(f"{cell_label:<12} SKIP — {pt_dir} missing")
        continue
    ckpt = get_latest_epoch(pt_dir)
    if ckpt is None:
        print(f"{cell_label:<12} SKIP — no checkpoint in {pt_dir}")
        continue
    latest_ep = int(ckpt.split("duorec-")[1].split(".pth")[0])

    try:
        model, rt_model, args = build_model(pt_dir, ckpt, is_notype)
    except Exception as e:
        print(f"{cell_label:<12} FAIL: {e}"); import traceback; traceback.print_exc(); continue

    pt_total, _ = count_params(model)
    rt_total, _ = count_params(rt_model) if rt_model else (0, 0)
    arch = "no-type" if is_notype else "content (type embeddings)"
    row = f"{cell_label:<12} ep{latest_ep:<38} {pt_total:>12,} {rt_total:>12,} {pt_total+rt_total:>12,}   {arch}"
    print(row)

    # Per-table breakdown for the first PACER-Full (author/music embeddings too)
    if cell_label == "PACER-Full":
        print("  ── PT per-table breakdown ──")
        for name, p in sorted(breakdown(model).items(), key=lambda kv: -kv[1]):
            print(f"    {name:<30} {p:>10,}")
        if rt_model:
            print("  ── RT per-table breakdown ──")
            for name, p in sorted(breakdown(rt_model).items(), key=lambda kv: -kv[1]):
                print(f"    {name:<30} {p:>10,}")

    del model, rt_model
    import gc; gc.collect()

print()
print("="*90)
print("All numbers = sum(p.numel() for p in model.parameters()).")
print("Author/music/duration embeddings only exist if -author_file/-music_file/-dur_file passed.")
print("so author_embedding.weight only appears in full-model checkpoints.")
