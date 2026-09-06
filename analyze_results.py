#!/usr/bin/env python3
"""
Analyze and compile all evaluation results into tables (console, CSV, LaTeX).

Auto-discovers every result file in the project directory:

  save_duorec_<variant>/test_result.txt            DuoRec, big-matrix protocol
  save_duorec_<variant>/test_result_small.txt      DuoRec, small-matrix protocol
  save_pt_fixrt_<config>_<variant>/test_result*.txt     TRIER with type embeddings
  save_pt_notype_fixrt_<config>_<variant>/test_result*.txt  TRIER without types
  baseline_results_<variant>/sasrec_results.txt         SASRec, big
  baseline_results_<variant>/sasrec_results_small.txt   SASRec, small
  baseline_results_<variant>/gru4rec_results.txt        GRU4Rec, big
  baseline_results_<variant>/gru4rec_results_small.txt  GRU4Rec, small

Configs (lambda sweep): nodiv, lamb0002, lamb0005, lamb0005_consec0001,
lamb001, lamb005, lamb01.

Usage:
    python3 analyze_results.py            # console table + results_summary.csv + results_tables.tex
    python3 analyze_results.py --proto small   # only small-matrix protocol
"""

import os
import ast
import re
import glob
import argparse
from collections import OrderedDict

# =============================================================================
# Configuration
# =============================================================================
VARIANTS = [
    ("kuairec_highest_individual", "Highest-Individual"),
    ("kuairec_highest_average", "Highest-Average"),
    ("kuairec_first_individual", "First-Individual"),
    ("kuairec_first_average", "First-Average"),
]
VARIANT_KEYS = [v[0] for v in VARIANTS]

# model family ordering for display
FAMILY_ORDER = {"baseline": 0, "duorec": 1, "trier_notype": 2, "trier_type": 3, "gru_trier": 4}

# config suffix -> display label (lambda sweep)
CONFIG_LABELS = {
    "nodiv": "No-Div",
    "lamb0002": "$\\lambda$=0.002",
    "lamb0005": "$\\lambda$=0.005",
    "lamb0005_consec0001": "$\\lambda$=0.005+Cons",
    "lamb001": "$\\lambda$=0.01",
    "lamb005": "$\\lambda$=0.05",
    "lamb01": "$\\lambda$=0.1",
}
CONFIG_ORDER = ["nodiv", "lamb0002", "lamb0005", "lamb0005_consec0001",
                "lamb001", "lamb005", "lamb01"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Parsing
# =============================================================================
def parse_dict_result(filepath):
    """Parse a test_result*.txt dict-literal file -> dict (or None)."""
    if not os.path.exists(filepath):
        return None
    try:
        content = open(filepath).read().strip()
        if not content:
            return None
        content = re.sub(r'np\.float\d*\(([^)]+)\)', r'\1', content)
        return ast.literal_eval(content)
    except Exception as e:
        print(f"  Warning: failed to parse {filepath}: {e}")
        return None


def parse_baseline_result(filepath):
    """Parse a key:value baseline results file -> dict (or None)."""
    if not os.path.exists(filepath):
        return None
    result = {}
    try:
        for line in open(filepath):
            m = re.match(r'\s*(Recall|MRR|NDCG|ILD|CS|CC)@(\d+):\s*([-\d.]+)', line)
            if m:
                result[f'{m.group(1).lower()}@{m.group(2)}'] = float(m.group(3))
    except Exception as e:
        print(f"  Warning: failed to parse {filepath}: {e}")
        return None
    return result or None


def split_dir_name(dirname):
    """Split a save_<prefix>_<rest> dir name into (family, config, variant_key).

    Recognized:
      save_duorec_<variant>
      save_pt_fixrt_<config>_<variant>
      save_pt_notype_fixrt_<config>_<variant>
      save_pt_gru_<config>_<variant>   (GRU-TRIER, if present)
    Returns None if unrecognized.
    """
    name = dirname[len("save_"):] if dirname.startswith("save_") else dirname

    if name.startswith("duorec_"):
        rest = name[len("duorec_"):]
        if rest in VARIANT_KEYS:
            return ("duorec", "duorec", rest)

    for prefix, family in (("pt_notype_fixrt_", "trier_notype"),
                           ("pt_fixrt_", "trier_type"),
                           ("pt_gru_", "gru_trier")):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            # config = everything before the variant suffix
            for vk in VARIANT_KEYS:
                if rest.endswith("_" + vk):
                    config = rest[: -(len(vk) + 1)]
                    return (family, config, vk)
    return None


def collect_all_results(proto_filter=None):
    """Return list of dicts: {family, config, variant, proto, label, data}."""
    rows = []

    # ---- TRIER / DuoRec dict-literal result files ----
    # Filenames:
    #   test_result.txt             big matrix;  duorec=topk, TRIER dirs=greedy
    #   test_result_small.txt       small matrix; duorec=topk, TRIER dirs=greedy
    #   test_result_topk.txt        big matrix;  topk full-catalog ranking
    #   test_result_topk_small.txt  small matrix; topk full-catalog ranking
    for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "save_*", "test_result*.txt"))):
        dirname = os.path.basename(os.path.dirname(path))
        basename = os.path.basename(path)
        if basename == "test_result.txt":
            proto, fname_infer = "big", None
        elif basename == "test_result_small.txt":
            proto, fname_infer = "small", None
        elif basename == "test_result_topk.txt":
            proto, fname_infer = "big", "topk"
        elif basename == "test_result_topk_small.txt":
            proto, fname_infer = "small", "topk"
        else:
            continue  # ignore test_result_500.txt etc.
        if proto_filter and proto != proto_filter:
            continue
        parsed = split_dir_name(dirname)
        if parsed is None:
            continue
        family, config, variant = parsed
        # inference mode: explicit from filename, else duorec is topk, TRIER greedy
        infer = fname_infer or ("topk" if family == "duorec" else "greedy")
        data = parse_dict_result(path)
        if data is None:
            continue
        rows.append({"family": family, "config": config, "variant": variant,
                     "proto": proto, "infer": infer, "data": data})

    # ---- Baselines (key:value text files) ----
    for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "baseline_results_*", "*_results*.txt"))):
        dirname = os.path.basename(os.path.dirname(path))  # baseline_results_<variant>
        variant = dirname[len("baseline_results_"):]
        if variant not in VARIANT_KEYS:
            continue
        basename = os.path.basename(path)                 # sasrec_results.txt / sasrec_results_small.txt
        m = re.match(r'^(sasrec|gru4rec|bert4rec)_results(_small)?\.txt$', basename)
        if not m:
            continue
        model = m.group(1)
        proto = "small" if m.group(2) else "big"
        if proto_filter and proto != proto_filter:
            continue
        data = parse_baseline_result(path)
        if data is None:
            continue
        rows.append({"family": "baseline", "config": model, "variant": variant,
                     "proto": proto, "infer": "topk", "data": data})

    return rows


# =============================================================================
# Metric access
# =============================================================================
# TRIER/dict keys vs baseline keys
METRIC_ALIASES = {
    "recall@5": ["recall@5_f", "recall@5"],
    "recall@10": ["recall@10_f", "recall@10"],
    "recall@20": ["recall@20_f", "recall@20"],
    "mrr@5": ["mrr@5_f", "mrr@5"],
    "mrr@10": ["mrr@10_f", "mrr@10"],
    "mrr@20": ["mrr@20_f", "mrr@20"],
    "ndcg@5": ["ndcg@5_f", "ndcg@5"],
    "ndcg@10": ["ndcg@10_f", "ndcg@10"],
    "ndcg@20": ["ndcg@20_f", "ndcg@20"],
    "ILD@5": ["ILD@5", "ild@5"],
    "ILD@10": ["ILD@10", "ild@10"],
    "ILD@20": ["ILD@20", "ild@20"],
    "CS@5": ["CS@5", "cs@5"],
    "CS@10": ["CS@10", "cs@10"],
    "CS@20": ["CS@20", "cs@20"],
    "CC@5": ["CC@5", "cc@5"],
    "CC@10": ["CC@10", "cc@10"],
    "CC@20": ["CC@20", "cc@20"],
}


def get_metric(data, metric):
    if data is None:
        return None
    for k in METRIC_ALIASES.get(metric, [metric]):
        if k in data and data[k] is not None:
            return data[k]
    return None


def model_label(row):
    fam, cfg = row["family"], row["config"]
    if fam == "baseline":
        return {"sasrec": "SASRec", "gru4rec": "GRU4Rec", "bert4rec": "BERT4Rec"}.get(cfg, cfg)
    if fam == "duorec":
        return "DuoRec"
    if fam == "gru_trier":
        return "GRU-TRIER " + CONFIG_LABELS.get(cfg, cfg)
    prefix = "TRIER(type) " if fam == "trier_type" else "TRIER(notype) "
    return prefix + CONFIG_LABELS.get(cfg, cfg)


def sort_key(row):
    fam_order = FAMILY_ORDER.get(row["family"], 9)
    if row["family"] == "baseline":
        cfg_order = {"sasrec": 0, "gru4rec": 1, "bert4rec": 2}.get(row["config"], 9)
    elif row["family"] == "duorec":
        cfg_order = -1
    else:
        cfg_order = CONFIG_ORDER.index(row["config"]) if row["config"] in CONFIG_ORDER else 9
    var_order = VARIANT_KEYS.index(row["variant"]) if row["variant"] in VARIANT_KEYS else 9
    return (fam_order, cfg_order, var_order)


# =============================================================================
# Output: console table
# =============================================================================
def print_console_table(rows, proto):
    proto_rows = [r for r in rows if r["proto"] == proto]
    if not proto_rows:
        print(f"\n(no {proto}-matrix results found)")
        return
    proto_rows.sort(key=sort_key)

    cols = [("R@10", "recall@10"), ("R@20", "recall@20"), ("N@10", "ndcg@10"),
            ("ILD@10", "ILD@10"), ("CS@10", "CS@10"), ("CC@10", "CC@10")]
    header = f"{'Model':<24} {'Infer':<7} {'Variant':<20} " + " ".join(f"{c[0]:>8}" for c in cols)
    print("\n" + "=" * len(header))
    print(f"{proto.upper()}-MATRIX PROTOCOL")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in proto_rows:
        label = model_label(r).replace("$\\lambda$", "λ")
        vals = []
        for _, m in cols:
            v = get_metric(r["data"], m)
            vals.append(f"{v:8.4f}" if v is not None else f"{'--':>8}")
        var_short = r["variant"].replace("kuairec_", "")
        print(f"{label:<24} {r.get('infer','?'):<7} {var_short:<20} " + " ".join(vals))


# =============================================================================
# Output: CSV
# =============================================================================
def write_csv(rows, path):
    metric_names = ["recall@5", "recall@10", "recall@20",
                    "mrr@5", "mrr@10", "mrr@20",
                    "ndcg@5", "ndcg@10", "ndcg@20",
                    "ILD@5", "ILD@10", "ILD@20",
                    "CS@5", "CS@10", "CS@20",
                    "CC@5", "CC@10", "CC@20"]
    with open(path, "w") as f:
        f.write("protocol,family,config,variant," + ",".join(metric_names) + "\n")
        for r in sorted(rows, key=sort_key):
            vals = []
            for m in metric_names:
                v = get_metric(r["data"], m)
                vals.append(f"{v:.6f}" if v is not None else "")
            f.write(",".join([r["proto"], r["family"], r["config"], r["variant"]] + vals) + "\n")
    print(f"CSV written: {path}")


# =============================================================================
# Output: LaTeX summary table
# =============================================================================
def tex_escape(s):
    return s

def write_latex(rows, path):
    cols = [("R@10", "recall@10"), ("N@10", "ndcg@10"),
            ("ILD@10", "ILD@10"), ("CS@10", "CS@10")]

    lines = [r"% Auto-generated by analyze_results.py", ""]
    for proto in ("big", "small"):
        proto_rows = [r for r in rows if r["proto"] == proto]
        if not proto_rows:
            continue

        # group rows by model (family+config), preserving variant order
        models = OrderedDict()
        for r in sorted(proto_rows, key=sort_key):
            key = (r["family"], r["config"])
            models.setdefault(key, {})[r["variant"]] = r["data"]

        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\caption{Results on " + proto + r"-matrix protocol "
                     r"(R@10, N@10, ILD@10, CS@10)}")
        lines.append(r"\label{tab:summary_" + proto + r"}")
        lines.append(r"\begin{tabular}{l" + "c" * (len(VARIANTS) * len(cols)) + r"}")
        lines.append(r"\toprule")

        h1 = r"\multirow{2}{*}{Model}"
        for i, (_, vl) in enumerate(VARIANTS):
            h1 += r" & \multicolumn{" + str(len(cols)) + r"}{c}{" + vl.replace("kuairec_", "").replace("_", "-") + r"}"
        h1 += r" \\"
        lines.append(h1)
        cmid = " ".join(
            r"\cmidrule(lr){" + str(2 + i * len(cols)) + "-" + str(1 + (i + 1) * len(cols)) + "}"
            for i in range(len(VARIANTS)))
        lines.append(cmid)
        lines.append(" & " + " & ".join(c[0] for c in cols * len(VARIANTS)) + r" \\")
        lines.append(r"\midrule")

        for (fam, cfg, infer), var_data in models.items():
            dummy = {"family": fam, "config": cfg}
            row = model_label(dummy)
            if fam not in ("baseline", "duorec"):
                row += " (" + infer + ")"
            for vk, _ in VARIANTS:
                data = var_data.get(vk)
                for _, m in cols:
                    v = get_metric(data, m)
                    row += " & " + (f"{v:.4f}" if v is not None else "--")
            row += r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX tables written: {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", choices=["big", "small"], default=None,
                        help="Only show one protocol (default: both)")
    args = parser.parse_args()

    print("Collecting results...")
    rows = collect_all_results(proto_filter=args.proto)
    print(f"Found {len(rows)} result files\n")

    if not rows:
        print("No results found. Run eval scripts first "
              "(eval_duorec.sh, eval_small_fixrt.sh, eval_greedy_fixrt.sh, eval_small_baselines.sh).")
        return

    protos = [args.proto] if args.proto else ["big", "small"]
    for p in protos:
        print_console_table(rows, p)

    write_csv(rows, os.path.join(SCRIPT_DIR, "results_summary.csv"))
    write_latex(rows, os.path.join(SCRIPT_DIR, "results_tables.tex"))


if __name__ == "__main__":
    main()
