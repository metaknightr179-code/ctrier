#!/usr/bin/env python3
"""Measure true parameter counts, one-forward-pass flops, and GPU memory for
the six-cell checkpoints. Run ONCE on AutoDL after pull — ~30 seconds.

Usage:  cd ~/ctrier && python3 measure_model_complexity.py
Output:  prints one row per checkpoint with real numbers; writes
         figures/complexity_actual_measured.txt so you can paste into the paper.
"""
import os, glob, sys, time, torch
import torch.nn as nn

sys.path.insert(0, ".")
from script import get_args, get_cates_map  # noqa
import main_pt  # noqa: defines TRIER_PT

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

VAR = "kuairec_first_average"
DATA_DIR = f"./KuaiRec_variants/{VAR}"
VEC = "./KuaiRec_variants/kuairec_vec.npy"
CATE_FILE = f"{DATA_DIR}/kuairec_cate.txt"
NEG_BIG = f"{DATA_DIR}/KuaiRec-random-sample_size=99-seed=4444.txt"
N = 10728; NCAT = 31
RT_DIR = f"save_rt_fix_{VAR}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}, GPU={torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_latest_epoch(pt_dir):
    files = sorted(glob.glob(f"{pt_dir}/model/duorec-*.pth"),
                   key=lambda p: int(p.split("duorec-")[1].split(".pth")[0]))
    return files[-1] if files else None

def build_model(pt_dir, latest_ckpt, type_flag):
    """Instantiate TRIER_PT, load checkpoint, set side-info maps (cates, etc.)."""
    args = get_args()
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
    args.no_type = bool(type_flag)  # -no_type → True
    args.dense = True                # our checkpoints are dense

    model = main_pt.TRIER_PT(N, args.ln, args.hn, args.hd, args.dr, args.b, args)
    # Load RT
    rt_ckpts = sorted(glob.glob(f"{RT_DIR}/model/duorec-*.pth"),
                       key=lambda p: int(p.split("duorec-")[1].split(".pth")[0]))
    if rt_ckpts:
        rt_epoch = int(rt_ckpts[-1].split("duorec-")[1].split(".pth")[0])
        rt_model = main_pt.TRIER_RT(N, 2, args.hn, args.hd, args.dr, args.b, args)
        rt_ckpt = rt_ckpts[-1]
        # Load rt_model (this is a separate model class — need its .load_state_dict)
        try:
            rt_state = torch.load(rt_ckpt, map_location=device)
            rt_model.load_state_dict(rt_state)
            rt_model.requires_grad_(False).eval()
            rt_model.to(device)
        except Exception as e:
            print(f"  RT load warn: {e}")
            rt_model = None
    else:
        rt_model = None

    try:
        state = torch.load(latest_ckpt, map_location=device)
        # Use the compat loader from main_pt
        from script import load_state_dict_compat
        load_state_dict_compat(model, latest_ckpt, device)
    except Exception as e:
        print(f"  PT load warn: {e}")

    model.requires_grad_(False).eval()
    model.to(device)

    # Load item categories into model
    try:
        cate_map = get_cates_map(CATE_FILE)
        model.set_item_types(cate_map)
    except Exception as e:
        print(f"  cate load warn: {e}")

    return model, rt_model, args

# Six-cell configs
CELLS = [
    ("TRIER",      "save_pt_notype_dense_lamb001_order0",  True),
    ("TRIER-C",    "save_pt_dense_lamb001_order0",        False),
    ("TRIER-L",    "save_pt_notype_dense_lamb001_softo001", True),
    ("TRIER-S",    "save_pt_notype_dense_lamb001_order0", True),
    ("PACER-LS",   "save_pt_notype_dense_lamb001_softo001", True),
    ("PACER-Full", "save_pt_dense_lamb001_softo001",       False),
]

def time_forward(model, rt_model, batch_size=32, warmup=3, runs=5):
    """Time one greedy decoding forward pass on CUDA."""
    if device.type != "cuda":
        return None
    with torch.no_grad():
        fake_session = torch.randint(1, N, (batch_size, 50), device=device)
        fake_reverse = torch.randint(1, N, (batch_size, 50), device=device)
        for _ in range(warmup):
            try:
                _, out = model.test_forward(fake_session, fake_reverse, rt_model, True)
            except Exception as e:
                print(f"  forward fail: {e}"); return None
        torch.cuda.synchronize()
        times = []
        t0 = time.time()
        for _ in range(runs):
            _, out = model.test_forward(fake_session, fake_reverse, rt_model, True)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / runs * 1000
    return elapsed  # ms per forward (batch_size sessions)

print()
print("="*90)
print(f"{'cell':<12} {'PT dir (latest epoch)':<45} {'PT params':>12} {'RT params':>12} {'PT+RT total':>12} {'fwd_ms/bs32':>12} {'peak_mem_MB':>12}")
print("="*90)

rows = []
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
        print(f"{cell_label:<12} BUILD FAIL: {e}"); continue

    pt_total, pt_train = count_params(model)
    rt_total, rt_train = count_params(rt_model) if rt_model else (0, 0)
    elapsed = time_forward(model, rt_model)

    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    row = f"{cell_label:<12} {pt_suffix+' ep'+str(latest_ep):<45} {pt_total:>12,} {rt_total:>12,} {pt_total+rt_total:>12,} {elapsed or 0:>12.1f} {peak_mem:>12.1f}"
    print(row)
    rows.append(row)

    # Free GPU memory before next model
    del model, rt_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

print()
print("="*90)
print("Run `free -h` on AutoDL after — GPU memory usage during each forward is the peak column above.")
print("The 'params' column is the ACTUAL count from sum(p.numel() for p in model.parameters()).")
print("You can paste those numbers into complexity_section.tex verbatim.")
print("="*90)
