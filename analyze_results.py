#!/usr/bin/env python3
"""
Analyze and compile all evaluation results into LaTeX tables.

Reads test_result.txt files from all PT configs and baselines,
produces accuracy and diversity tables for all variants.

Usage:
    python3 analyze_results.py
"""

import os
import ast
import re
import sys
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

PT_CONFIGS = [
    ("nodiv", 0.0, 0, "No-Div"),
    ("lamb0005", 0.005, 0, "$\\lambda=0.005$"),
    ("lamb001", 0.01, 0, "$\\lambda=0.01$"),
    ("lamb005", 0.05, 0, "$\\lambda=0.05$"),
    ("lamb01", 0.1, 0, "$\\lambda=0.1$"),
    ("consec0001", 0.01, 0.001, "$\\lambda=0.01$+Cons"),
]

GRU_CONFIGS = [
    ("nodiv", "GRU-No-Div"),
    ("lamb0005", "GRU $\\lambda=0.005$"),
    ("lamb001", "GRU $\\lambda=0.01$"),
    ("lamb005", "GRU $\\lambda=0.05$"),
    ("lamb01", "GRU $\\lambda=0.1$"),
    ("consec0001", "GRU $\\lambda=0.01$+Cons"),
]

BASELINES = [
    ("GRU4Rec", "gru4rec"),
    ("SASRec", "sasrec"),
    ("BERT4Rec", "bert4rec"),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def safe_parse_result(filepath):
    """Parse a test_result.txt file, return dict or None."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        if not content:
            return None
        # Replace np.float64(...) and np.float32(...) with just the value
        content = re.sub(r'np\.float64\(([^)]+)\)', r'\1', content)
        content = re.sub(r'np\.float32\(([^)]+)\)', r'\1', content)
        content = re.sub(r'np\.float\(([^)]+)\)', r'\1', content)
        data = ast.literal_eval(content)
        return data
    except Exception as e:
        print(f"  Warning: Failed to parse {filepath}: {e}")
        return None


def parse_baseline_result(filepath):
    """Parse a baseline results file, return dict or None."""
    if not os.path.exists(filepath):
        return None
    result = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' not in line or line.startswith(('GRU4Rec', 'SASRec', 'BERT4Rec', 'Training', 'Epochs')):
                    continue
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                try:
                    result[key.lower().replace(' ', '')] = float(val)
                except ValueError:
                    pass
    except Exception as e:
        print(f"  Warning: Failed to parse {filepath}: {e}")
        return None
    return result if result else None


def fmt(val, precision=4):
    """Format a float value, handle None."""
    if val is None:
        return "--"
    if isinstance(val, str):
        return val
    return f"{val:.{precision}f}"


def get_metric(data, key):
    """Extract a metric value from result dict."""
    if data is None:
        return None
    return data.get(key, None)


def collect_all_results():
    """Collect all results into nested dict: [model_key][variant_key] = data."""
    all_results = OrderedDict()

    # PT configs
    for suffix, lamb, lmd_consec, label in PT_CONFIGS:
        model_key = f"pt_{suffix}"
        all_results[model_key] = {"label": label, "results": {}}
        for var_key, var_label in VARIANTS:
            path = os.path.join(SCRIPT_DIR, f"save_pt_{suffix}_{var_key}", "test_result.txt")
            print(f"  PT {suffix} / {var_key}: {path}")
            data = safe_parse_result(path)
            all_results[model_key]["results"][var_key] = data

    # GRU-based TRIER configs (checkpoints under save_pt_gru_<suffix>_<variant>)
    for suffix, label in GRU_CONFIGS:
        model_key = f"gru_{suffix}"
        all_results[model_key] = {"label": label, "results": {}}
        for var_key, var_label in VARIANTS:
            path = os.path.join(SCRIPT_DIR, f"save_pt_gru_{suffix}_{var_key}", "test_result.txt")
            print(f"  GRU {suffix} / {var_key}: {path}")
            data = safe_parse_result(path)
            all_results[model_key]["results"][var_key] = data

    # Baselines
    for label, prefix in BASELINES:
        model_key = f"baseline_{prefix}"
        all_results[model_key] = {"label": label, "results": {}}
        for var_key, var_label in VARIANTS:
            path = os.path.join(SCRIPT_DIR, f"baseline_results_{var_key}", f"{prefix}_results.txt")
            print(f"  {label} / {var_key}: {path}")
            data = parse_baseline_result(path)
            all_results[model_key]["results"][var_key] = data

    return all_results


def generate_accuracy_table(all_results, var_key, var_label, prefix="pt_", label_prefix=""):
    """Generate LaTeX accuracy table for one variant."""
    metrics = [
        ("recall@5_f", "R@5"),
        ("recall@10_f", "R@10"),
        ("recall@20_f", "R@20"),
        ("mrr@5_f", "MRR@5"),
        ("mrr@10_f", "MRR@10"),
        ("mrr@20_f", "MRR@20"),
        ("ndcg@5_f", "N@5"),
        ("ndcg@10_f", "N@10"),
        ("ndcg@20_f", "N@20"),
    ]

    # Map baseline keys to our keys
    baseline_key_map = {
        "recall@5": "recall@5_f", "recall@10": "recall@10_f", "recall@20": "recall@20_f",
        "mrr@5": "mrr@5_f", "mrr@10": "mrr@10_f", "mrr@20": "mrr@20_f",
        "ndcg@5": "ndcg@5_f", "ndcg@10": "ndcg@10_f", "ndcg@20": "ndcg@20_f",
    }

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Accuracy Metrics on " + var_label + r"}")
    lines.append(r"\label{tab:" + label_prefix + r"acc_" + var_key.replace("kuairec_", "") + r"}")
    lines.append(r"\begin{tabular}{l" + "c" * len(metrics) + r"}")
    lines.append(r"\toprule")

    # Header
    header = "Model & " + " & ".join(m[1] for m in metrics) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best per metric (only among this family's models)
    best_vals = {}
    for m_key, m_label in metrics:
        best_vals[m_key] = -1
        for model_key, model_info in all_results.items():
            if not model_key.startswith(prefix):
                continue
            data = model_info["results"].get(var_key)
            if data:
                val = get_metric(data, m_key)
                if val is not None and val > best_vals[m_key]:
                    best_vals[m_key] = val

    # Model family rows
    for model_key, model_info in all_results.items():
        if not model_key.startswith(prefix):
            continue
        data = model_info["results"].get(var_key)
        if data is None:
            continue
        row_vals = []
        for m_key, m_label in metrics:
            val = get_metric(data, m_key)
            formatted = fmt(val)
            # Bold if best (within PT models, tolerance 1e-4)
            if val is not None and abs(val - best_vals[m_key]) < 1e-4:
                formatted = r"\textbf{" + formatted + r"}"
            row_vals.append(formatted)
        line = model_info["label"] + " & " + " & ".join(row_vals) + r" \\"
        lines.append(line)

    lines.append(r"\midrule")

    # Baseline rows
    for model_key, model_info in all_results.items():
        if not model_key.startswith("baseline_"):
            continue
        data = model_info["results"].get(var_key)
        if data is None:
            continue
        row_vals = []
        for m_key, m_label in metrics:
            # Try direct key first, then mapped key
            val = get_metric(data, m_key)
            if val is None:
                mapped_key = baseline_key_map.get(m_key, m_key)
                val = get_metric(data, mapped_key)
            row_vals.append(fmt(val))
        line = model_info["label"] + " & " + " & ".join(row_vals) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_diversity_table(all_results, var_key, var_label, prefix="pt_", label_prefix=""):
    """Generate LaTeX diversity table for one variant."""
    metrics = [
        ("ILD@5", "ILD@5"),
        ("ILD@10", "ILD@10"),
        ("ILD@20", "ILD@20"),
        ("CS@5", "CS@5"),
        ("CS@10", "CS@10"),
        ("CS@20", "CS@20"),
        ("CC@5", "CC@5"),
        ("CC@10", "CC@10"),
        ("CC@20", "CC@20"),
    ]

    # Map baseline keys (lowercase, no @)
    baseline_key_map = {m[0].lower().replace("@", ""): m[0] for m in metrics}

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Diversity Metrics on " + var_label + r"}")
    lines.append(r"\label{tab:" + label_prefix + r"div_" + var_key.replace("kuairec_", "") + r"}")
    lines.append(r"\begin{tabular}{l" + "c" * len(metrics) + r"}")
    lines.append(r"\toprule")

    header = "Model & " + " & ".join(m[1] for m in metrics) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best per metric (for diversity: ILD higher=better, CS lower=better, CC higher=better)
    best_vals = {}
    for m_key, m_label in metrics:
        best_vals[m_key] = -1e9 if ("ILD" in m_key or "CC" in m_key) else 1e9
        for model_key, model_info in all_results.items():
            if not model_key.startswith(prefix):
                continue
            data = model_info["results"].get(var_key)
            if data:
                val = get_metric(data, m_key)
                if val is not None:
                    if "ILD" in m_key or "CC" in m_key:
                        if val > best_vals[m_key]:
                            best_vals[m_key] = val
                    else:  # CS: lower is better
                        if val < best_vals[m_key]:
                            best_vals[m_key] = val

    # Model family rows
    for model_key, model_info in all_results.items():
        if not model_key.startswith(prefix):
            continue
        data = model_info["results"].get(var_key)
        if data is None:
            continue
        row_vals = []
        for m_key, m_label in metrics:
            val = get_metric(data, m_key)
            formatted = fmt(val)
            if val is not None:
                if "ILD" in m_key or "CC" in m_key:
                    if abs(val - best_vals[m_key]) < 1e-3:
                        formatted = r"\textbf{" + formatted + r"}"
                else:
                    if abs(val - best_vals[m_key]) < 1e-3:
                        formatted = r"\textbf{" + formatted + r"}"
            row_vals.append(formatted)
        line = model_info["label"] + " & " + " & ".join(row_vals) + r" \\"
        lines.append(line)

    lines.append(r"\midrule")

    # Baseline rows
    for model_key, model_info in all_results.items():
        if not model_key.startswith("baseline_"):
            continue
        data = model_info["results"].get(var_key)
        if data is None:
            continue
        row_vals = []
        for m_key, m_label in metrics:
            val = get_metric(data, m_key)
            if val is None:
                mapped_key = baseline_key_map.get(m_key.lower().replace("@", ""))
                if mapped_key:
                    val = get_metric(data, mapped_key)
            row_vals.append(fmt(val))
        line = model_info["label"] + " & " + " & ".join(row_vals) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_summary_table(all_results, prefix="pt_", label="tab:summary",
                           caption="Summary: Accuracy and Diversity (R@10, N@10, ILD@10, CC@10) Across Variants",
                           include_baselines=True):
    """Generate a summary table across all variants (R@10, N@10, ILD@10, CC@10)."""
    metrics = [
        ("recall@10_f", "R@10", False),
        ("ndcg@10_f", "N@10", False),
        ("ILD@10", "ILD@10", True),  # higher better
        ("CC@10", "CC@10", True),    # higher better
    ]

    baseline_key_map = {
        "recall@10_f": "recall@10", "ndcg@10_f": "ndcg@10",
        "ILD@10": "ild@10", "CC@10": "cc@10",
    }

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    var_cols = "l" + "c" * (len(VARIANTS) * len(metrics))
    lines.append(r"\begin{tabular}{" + var_cols + r"}")
    lines.append(r"\toprule")

    # Multi-row header
    header1 = r"\multirow{2}{*}{Model} & "
    for i, (vk, vl) in enumerate(VARIANTS):
        header1 += r"\multicolumn{" + str(len(metrics)) + r"}{c}{" + vl + r"}"
        if i < len(VARIANTS) - 1:
            header1 += " & "
    header1 += r" \\"
    lines.append(header1)

    lines.append(r"\cmidrule(lr){2-" + str(1 + len(metrics)) + r"}" +
                 r" \cmidrule(lr){" + str(2 + len(metrics)) + "-" + str(1 + 2*len(metrics)) + r"}" +
                 r" \cmidrule(lr){" + str(2 + 2*len(metrics)) + "-" + str(1 + 3*len(metrics)) + r"}" +
                 r" \cmidrule(lr){" + str(2 + 3*len(metrics)) + "-" + str(1 + 4*len(metrics)) + r"}")

    header2 = " & " + " & ".join([m[1] for m in metrics] * len(VARIANTS)) + r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Model family rows
    for model_key, model_info in all_results.items():
        if not model_key.startswith(prefix):
            continue
        row = model_info["label"]
        for vk, vl in VARIANTS:
            data = model_info["results"].get(vk)
            for m_key, m_label, _ in metrics:
                val = get_metric(data, m_key) if data else None
                row += " & " + fmt(val)
        row += r" \\"
        lines.append(row)

    if include_baselines:
        lines.append(r"\midrule")

        # Baseline rows
        for model_key, model_info in all_results.items():
            if not model_key.startswith("baseline_"):
                continue
            row = model_info["label"]
            for vk, vl in VARIANTS:
                data = model_info["results"].get(vk)
                for m_key, m_label, _ in metrics:
                    val = None
                    if data:
                        val = get_metric(data, m_key)
                        if val is None:
                            mapped = baseline_key_map.get(m_key, m_key)
                            val = get_metric(data, mapped.lower().replace(" ", ""))
                    row += " & " + fmt(val)
            row += r" \\"
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_csv_summary(all_results):
    """Generate CSV summary for easy copy/paste."""
    lines = []
    header = "Model,Variant,R@5,R@10,R@20,MRR@5,MRR@10,MRR@20,NDCG@5,NDCG@10,NDCG@20,ILD@5,ILD@10,ILD@20,CS@5,CS@10,CS@20,CC@5,CC@10,CC@20"
    lines.append(header)

    metric_keys = [
        "recall@5_f", "recall@10_f", "recall@20_f",
        "mrr@5_f", "mrr@10_f", "mrr@20_f",
        "ndcg@5_f", "ndcg@10_f", "ndcg@20_f",
        "ILD@5", "ILD@10", "ILD@20",
        "CS@5", "CS@10", "CS@20",
        "CC@5", "CC@10", "CC@20",
    ]
    baseline_map = {
        "recall@5_f": "recall@5", "recall@10_f": "recall@10", "recall@20_f": "recall@20",
        "mrr@5_f": "mrr@5", "mrr@10_f": "mrr@10", "mrr@20_f": "mrr@20",
        "ndcg@5_f": "ndcg@5", "ndcg@10_f": "ndcg@10", "ndcg@20_f": "ndcg@20",
        "ILD@5": "ild@5", "ILD@10": "ild@10", "ILD@20": "ild@20",
        "CS@5": "cs@5", "CS@10": "cs@10", "CS@20": "cs@20",
        "CC@5": "cc@5", "CC@10": "cc@10", "CC@20": "cc@20",
    }

    for model_key, model_info in all_results.items():
        for vk, vl in VARIANTS:
            data = model_info["results"].get(vk)
            row = [model_info["label"], vl]
            for mk in metric_keys:
                val = None
                if data:
                    val = get_metric(data, mk)
                    if val is None:
                        bkey = baseline_map.get(mk, mk).lower().replace(" ", "")
                        val = get_metric(data, bkey)
                row.append(fmt(val))
            lines.append(",".join(row))

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Collecting results...")
    print("=" * 60)
    all_results = collect_all_results()

    # Count available results
    total = 0
    found = 0
    for mk, mi in all_results.items():
        for vk, vl in VARIANTS:
            total += 1
            if mi["results"].get(vk) is not None:
                found += 1
    print(f"\nFound {found}/{total} result files\n")

    if found == 0:
        print("No results found. Run eval_all.sh first.")
        return

    print("=" * 60)
    print("Generating LaTeX tables...")
    print("=" * 60)

    output = []
    output.append("% ============================================================")
    output.append("% Auto-generated LaTeX tables from evaluation results")
    output.append("% Generated by analyze_results.py")
    output.append("% ============================================================")
    output.append("")

    # Summary table (all variants in one)
    output.append("% --- Summary Table ---")
    output.append(generate_summary_table(all_results))
    output.append("")

    # Per-variant tables
    for vk, vl in VARIANTS:
        output.append(f"% --- {vl} ---")
        output.append(generate_accuracy_table(all_results, vk, vl))
        output.append("")
        output.append(generate_diversity_table(all_results, vk, vl))
        output.append("")

    # GRU-based TRIER tables (separate section)
    output.append("% --- GRU Summary Table ---")
    output.append(generate_summary_table(
        all_results, prefix="gru_", label="tab:gru_summary",
        caption="GRU-based TRIER: Accuracy and Diversity (R@10, N@10, ILD@10, CC@10) Across Variants"))
    output.append("")
    for vk, vl in VARIANTS:
        output.append(f"% --- GRU {vl} ---")
        output.append(generate_accuracy_table(all_results, vk, vl, prefix="gru_", label_prefix="gru_"))
        output.append("")
        output.append(generate_diversity_table(all_results, vk, vl, prefix="gru_", label_prefix="gru_"))
        output.append("")

    # CSV summary
    output.append("% --- CSV Summary ---")
    output.append("% (commented out — uncomment to use)")
    csv_lines = generate_csv_summary(all_results)
    for line in csv_lines.split("\n"):
        output.append("% " + line)
    output.append("")

    latex_content = "\n".join(output)

    # Write to file
    out_file = os.path.join(SCRIPT_DIR, "results_tables.tex")
    with open(out_file, 'w') as f:
        f.write(latex_content)
    print(f"\nLaTeX tables written to: {out_file}")

    # Also print to stdout
    print("\n" + "=" * 60)
    print(latex_content)
    print("=" * 60)

    # Also write CSV
    csv_file = os.path.join(SCRIPT_DIR, "results_summary.csv")
    with open(csv_file, 'w') as f:
        f.write(generate_csv_summary(all_results))
    print(f"\nCSV summary written to: {csv_file}")


if __name__ == "__main__":
    main()
