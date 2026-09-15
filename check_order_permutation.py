#!/usr/bin/env python3
# =============================================================================
# TOY PROOF — same Top-K item SET, three different ORDERINGS.
#
# Research question (slides: "同集合不同顺序"): set-level diversity metrics
# cannot see ORDER, while position-aware metrics can. Fix one Top-K set and
# permute it three ways:
#   blocked     - similar items clumped together:   a a a a ... b b b b ...
#   alternating - categories alternate:             a b a b a b ...
#   random      - a fixed-seed random permutation
#
# Verified on REAL KuaiRec content vectors/categories:
#   * ILD@k  (mean pairwise euclidean distance) is SET-ONLY -> identical
#   * CC@k   (category coverage) is SET-ONLY          -> identical
#   * CS@k   (mean adjacent cosine, script.py)        -> differs
#   * MaxRun@k (longest run sharing a cat / cos>0.7)  -> differs
#   * L_order (realized adjacent hinge, trier_pt.py)  -> differs
# L_order is the one-hot (fully sharpened soft-selection) realization of
# soft_order_loss: with pi_s a point mass on the actually picked item,
#   L_order = mean_s relu(cos(v_{pi_s}, v_{pi_{s+1}}) - 0.5),
# exactly the margin used in TRIER_PT.soft_order_loss.
#
# Item selection is deterministic: the two largest categories whose items
# carry that category EXCLUSIVELY are used, so within-group cos = 1 and
# cross-group cos = 0 analytically. The random ordering is the first seed in
# a fixed scan whose metrics land strictly between the two extremes.
#
# Usage:  python3 check_order_permutation.py
# Exit code 0 = PASS (ILD/CC invariant; CS/MaxRun/L_order ordered & different).
# =============================================================================
import os
import random

import numpy as np
import torch

VEC_PATH = "./KuaiRec_variants/kuairec_vec.npy"
CATE_PATH = "./KuaiRec_variants/kuairec_first_average/kuairec_cate.txt"
N_CAT = 31
MARGIN = 0.5          # soft_order_loss hinge margin (trier_pt.py)
TAU_RUN = 0.7         # MaxRun cosine threshold (script.py)
KS = (5, 10, 20)


def load_cate_map(path):
    """Mirror script.get_cates_map: iid -> [cate_id, ...]."""
    cmap = {}
    with open(path) as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(" ") if x]
            if parts:
                cmap[parts[0]] = parts[1:]
    return cmap


def cal_ILD(vecs, k):
    """Exact replica of script.cal_ILD (torch cdist == sklearn euclidean)."""
    d = torch.cdist(vecs, vecs)
    return d.sum().item() / (k * (k - 1))


def sorted_pairwise_distances(vecs):
    """Sorted multiset of pairwise distances — order-invariant by construction."""
    d = torch.cdist(vecs, vecs)
    k = vecs.shape[0]
    off = d[~torch.eye(k, dtype=torch.bool)]
    return torch.sort(off)[0]


def adjacent_cos(vecs):
    return torch.nn.functional.cosine_similarity(vecs[:-1], vecs[1:], dim=-1)


def metric_CS(vecs):
    return adjacent_cos(vecs).mean().item()


def metric_L_order(vecs):
    """One-hot-pi realization of TRIER_PT.soft_order_loss (margin 0.5)."""
    return torch.relu(adjacent_cos(vecs) - MARGIN).mean().item()


def metric_MaxRun(items, vecs, cmap):
    """Exact replica of the MaxRun@k block in script.evaluate_function_with_full."""
    cos_sim = adjacent_cos(vecs)
    sim_cos = cos_sim > TAU_RUN
    k = len(items)
    sim_cat = torch.zeros(k - 1, dtype=torch.bool)
    for s in range(k - 1):
        c_prev = set(cmap.get(items[s], []))
        c_curr = set(cmap.get(items[s + 1], []))
        if c_prev and c_prev.intersection(c_curr):
            sim_cat[s] = True
    a_s = sim_cos.cpu() | sim_cat
    max_run, cur = 1, 1
    for s in range(k - 1):
        if a_s[s]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return float(max_run), a_s


def metric_CC(items, cmap):
    """script.coverage_for_user denominator equivalent: |union of cats|/n_cat."""
    cats = set()
    for it in items:
        cats.update(cmap.get(it, []))
    return len(cats) / N_CAT


def pick_two_groups(k, cmap):
    """Two disjoint single-category item groups, sizes m and k-m."""
    m = k // 2
    exclusive = {}                       # cat -> [items carrying ONLY that cat]
    for iid, cats in cmap.items():
        if len(cats) == 1:
            exclusive.setdefault(cats[0], []).append(iid)
    ranked = sorted(exclusive.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    (ca, items_a), (cb, items_b) = ranked[0], ranked[1]
    assert len(items_a) >= m and len(items_b) >= k - m, "not enough exclusive items"
    return sorted(items_a)[:m], sorted(items_b)[:k - m], ca, cb


def measure(name, items, vec_table, cmap, show_seq=False):
    vecs = vec_table[torch.tensor(items, dtype=torch.long)]
    k = len(items)
    maxrun, flags = metric_MaxRun(items, vecs, cmap)
    out = {
        "name": name,
        "ILD": cal_ILD(vecs, k),
        "CC": metric_CC(items, cmap),
        "CS": metric_CS(vecs),
        "MaxRun": maxrun,
        "L_order": metric_L_order(vecs),
    }
    if show_seq:
        seq = "".join("1" if x else "0" for x in flags.tolist())
        out["adj_similar"] = seq
        out["adj_cos"] = [f"{x:.2f}" for x in adjacent_cos(vecs).tolist()]
    return out


def main():
    assert os.path.exists(VEC_PATH), f"missing {VEC_PATH} (run from repo root)"
    assert os.path.exists(CATE_PATH), f"missing {CATE_PATH}"
    vec_table = torch.tensor(np.load(VEC_PATH), dtype=torch.float32)
    cmap = load_cate_map(CATE_PATH)

    all_rows = {}
    passes = []
    for k in KS:
        ga, gb, ca, cb = pick_two_groups(k, cmap)
        blocked = ga + gb
        # Interleave starting with the LARGER group so odd-k lists end with it
        # too (k=5: b a b a b -> zero adjacent-similar pairs).
        big_g, small_g = (gb, ga) if len(gb) >= len(ga) else (ga, gb)
        alternating = []
        for i in range(len(big_g)):
            alternating.append(big_g[i])
            if i < len(small_g):
                alternating.append(small_g[i])

        # deterministic "random": first seed strictly between the extremes on
        # the continuous metrics CS and L_order (at small/odd k, MaxRun is an
        # integer with possibly no value strictly between the two extremes, so
        # it is only required to stay within the extreme bracket).
        rnd = None
        for seed in range(100):
            cand = blocked[:]
            random.Random(seed).shuffle(cand)
            r = measure("random", cand, vec_table, cmap)
            b0 = measure("b", blocked, vec_table, cmap)
            a0 = measure("a", alternating, vec_table, cmap)
            if (a0["CS"] < r["CS"] < b0["CS"] and
                    a0["L_order"] < r["L_order"] < b0["L_order"] and
                    a0["MaxRun"] <= r["MaxRun"] <= b0["MaxRun"]):
                rnd, rnd_seed = cand, seed
                break
        assert rnd is not None, "no intermediate random seed found"

        rows = [
            measure("blocked", blocked, vec_table, cmap, show_seq=(k == 20)),
            measure("alternating", alternating, vec_table, cmap, show_seq=(k == 20)),
            measure("random", rnd, vec_table, cmap, show_seq=(k == 20)),
        ]
        all_rows[k] = (rows, (ca, cb), rnd_seed)

        # ---- assertions ---------------------------------------------------
        ilds = [r["ILD"] for r in rows]
        ccs = [r["CC"] for r in rows]
        # pairwise-distance MULTISETS are exactly identical (strongest check)
        items_by_name = {"blocked": blocked, "alternating": alternating, "random": rnd}
        d_sorted = [sorted_pairwise_distances(
                        vec_table[torch.tensor(items_by_name[r["name"]],
                                               dtype=torch.long)])
                    for r in rows]
        exact_set_eq = all(torch.equal(d_sorted[0], d) for d in d_sorted[1:])
        # ILD is the SUM of that identical distance multiset; only float32
        # reduction order can differ across permutations.
        ild_spread = max(ilds) - min(ilds)
        ild_eq = ild_spread < 1e-6
        cc_eq = len(set(ccs)) == 1
        b, a, r = rows[0], rows[1], rows[2]
        # extremes must be strictly separated on EVERY order metric; the random
        # row must be strictly intermediate on the continuous ones (CS, L_order)
        extremes_separated = (b["CS"] > a["CS"] and
                              b["MaxRun"] > a["MaxRun"] and
                              b["L_order"] > a["L_order"])
        ordered = (b["CS"] > r["CS"] > a["CS"] and
                   b["L_order"] > r["L_order"] > a["L_order"] and
                   b["MaxRun"] >= r["MaxRun"] >= a["MaxRun"])
        ok = exact_set_eq and ild_eq and cc_eq and extremes_separated and ordered
        passes.append(ok)

        print(f"--- K={k}  (cats A={ca}, B={cb}; |A|={len(ga)}, |B|={len(gb)}; "
              f"random seed={rnd_seed})")
        print(f"{'ordering':<12} {'ILD@k':>10} {'CC@k':>8} {'CS@k':>10} "
              f"{'MaxRun@k':>9} {'L_order':>10}")
        for row in rows:
            print(f"{row['name']:<12} {row['ILD']:>10.6f} {row['CC']:>8.4f} "
                  f"{row['CS']:>10.6f} {row['MaxRun']:>9.0f} {row['L_order']:>10.6f}")
        print(f"  pairwise-distance multisets identical: {exact_set_eq}; "
              f"ILD spread (float32 reduction order): {ild_spread:.2e}; "
              f"CC equal: {cc_eq}; "
              f"CS/MaxRun/L ordered blocked>random>alternating: {ordered}")
        if k == 20:
            for row in rows:
                print(f"  [{row['name']:<11}] adjacent-similar flags: {row['adj_similar']}")
        print()

    # ---- paper-ready LaTeX fragment (K=20) ---------------------------------
    rows20, _, seed20 = all_rows[20]
    label = {"blocked": "Clumped", "alternating": "Alternating", "random": "Random"}
    print("LaTeX fragment (paste into the main-text proof table):")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Ordering & ILD@20 (same) & CS@20 $\downarrow$ & MaxRun@20 $\downarrow$ "
          r"& $\mathcal{L}_{order}$ $\downarrow$ \\")
    print(r"\midrule")
    for row in rows20:
        print(f"{label[row['name']]} & {row['ILD']:.4f} & {row['CS']:.4f} & "
              f"{row['MaxRun']:.0f} & {row['L_order']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"% random permutation uses random.Random({seed20}).shuffle; "
          f"CC@20 = {rows20[0]['CC']:.4f} for all three orderings")

    print()
    print("RESULT:", "PASS" if all(passes) else "FAIL")
    raise SystemExit(0 if all(passes) else 1)


if __name__ == "__main__":
    main()
