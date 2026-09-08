#!/usr/bin/env python3
"""
Analyze baseline (SASRec, GRU4Rec, BERT4Rec) results across all datasets.

Auto-discovers baseline_results_<variant>/ directories for:
  - KuaiRec variants: kuairec_highest_individual, kuairec_highest_average,
                      kuairec_first_individual, kuairec_first_average
  - New datasets: ML1M, KuaiRand1K, MicroLens

Parses key:value files like sasrec_results.txt, gru4rec_results.txt.

Usage:
    python3 analyze_baselines.py                    # console table
    python3 analyze_baselines.py --proto small      # small-matrix only
    python3 analyze_baselines.py --csv baselines.csv
    python3 analyze_baselines.py --tex baselines.tex
"""

import os
import re
import glob
import argparse
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# All known variant/dataset names (KuaiRec splits + new datasets)
DATASETS = [
    ("kuairec_highest_individual", "KuaiRec-HI"),
    ("kuairec_highest_average", "KuaiRec-HA"),
    ("kuairec_first_individual", "KuaiRec-FI"),
    ("kuairec_first_average", "KuaiRec-FA"),
    ("ML1M", "ML-1M"),
    ("KuaiRand1K", "KuaiRand"),
    ("MicroLens", "MicroLens"),
]
DATASET_KEYS = [d[0] for d in DATASETS]

MODELS = ["sasrec", "gru4rec", "bert4rec"]
MODEL_LABELS = {"sasrec": "SASRec", "gru4rec": "GRU4Rec", "bert4rec": "BERT4Rec"}


def parse_baseline_result(filepath):
    """Parse a key:value baseline results file -> dict (or None)."""
    if not os.path.exists(filepath):
        return None
    result = {}
    try:
        for line in open(filepath):
            m = re.match(r'\s*(Recall|MRR|NDCG|ILD|CS|CC)@(\d+):\s*([-\d.]+)', line)
            if m:
                key = f'{m.group(1).lower()}@{m.group(2)}'
                result[key] = float(m.group(3))
    except Exception as e:
        print(f"  Warning: failed to parse {filepath}: {e}")
        return None
    return result or None


def collect_baseline_results(proto_filter=None):
    """Return list of dicts: {model, variant, proto, data}."""
    rows = []
    for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "baseline_results_*", "*_results*.txt"))):
        dirname = os.path.basename(os.path.dirname(path))  # baseline_results_<variant>
        variant = dirname[len("baseline_results_"):]
        if variant not in DATASET_KEYS:
            continue
        basename = os.path.basename(path)
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
        rows.append({"model": model, "variant": variant,
                     "proto": proto, "data": data})
    return rows


def print_console_table(rows, proto):
    proto_rows = [r for r in rows if r["proto"] == proto]
    if not proto_rows:
        print(f"\n(no {proto}-matrix baseline results found)")
        return

    # Group by dataset
    datasets_present = []
    for dk, _ in DATASETS:
        if any(r["variant"] == dk for r in proto_rows):
            datasets_present.append(dk)

    cols = [("R@10", "recall@10"), ("R@20", "recall@20"), ("N@10", "ndcg@10"),
            ("ILD@10", "ild@10"), ("CS@10", "cs@10"), ("CC@10", "cc@10")]
    header = f"{'Model':<10} {'Dataset':<25} " + " ".join(f"{c[0]:>8}" for c in cols)
    print("\n" + "=" * len(header))
    print(f"{proto.upper()}-MATRIX PROTOCOL — BASELINES ONLY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in sorted(proto_rows, key=lambda x: (DATASET_KEYS.index(x["variant"]),
                                              MODELS.index(x["model"]) if x["model"] in MODELS else 9)):
        label = MODEL_LABELS.get(r["model"], r["model"])
        ds_label = dict(DATASETS).get(r["variant"], r["variant"])
        vals = []
        for _, m in cols:
            v = r["data"].get(m)
            vals.append(f"{v:8.4f}" if v is not None else f"{'--':>8}")
        print(f"{label:<10} {ds_label:<25} " + " ".join(vals))


def write_csv(rows, path):
    metric_names = ["recall@5", "recall@10", "recall@20",
                    "mrr@5", "mrr@10", "mrr@20",
                    "ndcg@5", "ndcg@10", "ndcg@20",
                    "ild@5", "ild@10", "ild@20",
                    "cs@5", "cs@10", "cs@20",
                    "cc@5", "cc@10", "cc@20"]
    with open(path, "w") as f:
        f.write("protocol,model,variant," + ",".join(metric_names) + "\n")
        for r in sorted(rows, key=lambda x: (r["proto"], DATASET_KEYS.index(x["variant"]),
                                              MODELS.index(x["model"]) if x["model"] in MODELS else 9)):
            vals = []
            for m in metric_names:
                v = r["data"].get(m)
                vals.append(f"{v:.6f}" if v is not None else "")
            f.write(",".join([r["proto"], r["model"], r["variant"]] + vals) + "\n")
    print(f"CSV written: {path}")


def write_latex(rows, path):
    cols = [("R@10", "recall@10"), ("R@20", "recall@20"),
            ("N@10", "ndcg@10"), ("ILD@10", "ild@10")]

    lines = [r"% Auto-generated by analyze_baselines.py", ""]
    for proto in ("big", "small"):
        proto_rows = [r for r in rows if r["proto"] == proto]
        if not proto_rows:
            continue

        # Group by model, preserving dataset order
        models = OrderedDict()
        for r in sorted(proto_rows, key=lambda x: (MODELS.index(x["model"]) if x["model"] in MODELS else 9,
                                                     DATASET_KEYS.index(x["variant"]))):
            key = r["model"]
            models.setdefault(key, {})[r["variant"]] = r["data"]

        # Only include datasets that have at least one result
        datasets_present = []
        for dk, _ in DATASETS:
            if any(dk in m for m in models.values()):
                datasets_present.append((dk, dict(DATASETS)[dk]))

        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\caption{Baseline results on " + proto + r"-matrix protocol}")
        lines.append(r"\label{tab:baselines_" + proto + r"}")
        lines.append(r"\begin{tabular}{l" + "c" * (len(datasets_present) * len(cols)) + r"}")
        lines.append(r"\toprule")

        h1 = r"\multirow{2}{*}{Model}"
        for _, dl in datasets_present:
            h1 += r" & \multicolumn{" + str(len(cols)) + r"}{c}{" + dl + r"}"
        h1 += r" \\"
        lines.append(h1)
        cmid = " ".join(
            r"\cmidrule(lr){" + str(2 + i * len(cols)) + "-" + str(1 + (i + 1) * len(cols)) + "}"
            for i in range(len(datasets_present)))
        lines.append(cmid)
        lines.append(" & " + " & ".join(c[0] for c in cols * len(datasets_present)) + r" \\")
        lines.append(r"\midrule")

        for model, var_data in models.items():
            row = MODEL_LABELS.get(model, model)
            for dk, _ in datasets_present:
                data = var_data.get(dk)
                for _, m in cols:
                    v = data.get(m) if data else None
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


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline results")
    parser.add_argument("--proto", choices=["big", "small"], default=None,
                        help="Only show one protocol (default: both)")
    parser.add_argument("--csv", default=None, help="Write CSV to this path")
    parser.add_argument("--tex", default=None, help="Write LaTeX tables to this path")
    args = parser.parse_args()

    print("Collecting baseline results...")
    rows = collect_baseline_results(proto_filter=args.proto)
    print(f"Found {len(rows)} baseline result files\n")

    if not rows:
        print("No baseline results found. Run train_baselines_newds.sh / eval_small_baselines.sh first.")
        return

    protos = [args.proto] if args.proto else ["big", "small"]
    for p in protos:
        print_console_table(rows, p)

    csv_path = args.csv or os.path.join(SCRIPT_DIR, "baselines_summary.csv")
    write_csv(rows, csv_path)

    tex_path = args.tex or os.path.join(SCRIPT_DIR, "baselines_tables.tex")
    write_latex(rows, tex_path)


if __name__ == "__main__":
    main()
