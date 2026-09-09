#!/usr/bin/env python3
"""
Analyze and compile all evaluation results into tables (console, CSV, LaTeX).

Auto-discovers every result file in the project directory:

  save_duorec_<variant>/test_result.txt            DuoRec, big-matrix protocol
  save_duorec_<variant>/test_result_small.txt      DuoRec, small-matrix protocol
  save_pt_fixrt_<config>_<variant>/test_result*.txt     TRIER with type embeddings
  save_pt_notype_fixrt_<config>_<variant>/test_result*.txt  TRIER without types
  save_pt_dense_<config>_<dataset>/test_result*.txt     TRIER dense, type
  save_pt_notype_dense_<config>_<dataset>/test_result*.txt  TRIER dense, notype
  save_pt_<config>_<dataset>/test_result*.txt           TRIER non-dense, type (new datasets)
  save_pt_notype_<config>_<dataset>/test_result*.txt    TRIER non-dense, notype (new datasets)
  baseline_results_<variant>/sasrec_results.txt         SASRec, big
  baseline_results_<variant>/gru4rec_results.txt        GRU4Rec, big

Configs (lambda sweep): nodiv, lamb0002, lamb0005, lamb0005_consec0001,
lamb001, lamb005, lamb01.

Usage:
    python3 analyze_results.py                          # KuaiRec (default)
    python3 analyze_results.py --dataset ML1M           # ML-1M only
    python3 analyze_results.py --dataset KuaiRand1K      # KuaiRand only
    python3 analyze_results.py --dataset MicroLens       # MicroLens only
    python3 analyze_results.py --dataset all             # everything
    python3 analyze_results.py --proto small             # small-matrix only
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
DATASETS = {
    "kuairec": {
        "variants": [
            ("kuairec_highest_individual", "Highest-Individual"),
            ("kuairec_highest_average", "Highest-Average"),
            ("kuairec_first_individual", "First-Individual"),
            ("kuairec_first_average", "First-Average"),
        ],
        "label": "KuaiRec",
        "prefixes": {"type": "pt_fixrt_", "notype": "pt_notype_fixrt_"},
    },
    "ML1M": {
        "variants": [("ML1M", "ML-1M")],
        "label": "ML-1M",
        "prefixes": {"type_dense": "pt_dense_", "notype_dense": "pt_notype_dense_",
                     "type": "pt_", "notype": "pt_notype_"},
    },
    "KuaiRand1K": {
        "variants": [("KuaiRand1K", "KuaiRand")],
        "label": "KuaiRand",
        "prefixes": {"type_dense": "pt_dense_", "notype_dense": "pt_notype_dense_",
                     "type": "pt_", "notype": "pt_notype_"},
    },
    "MicroLens": {
        "variants": [("MicroLens", "MicroLens")],
        "label": "MicroLens",
        "prefixes": {"type_dense": "pt_dense_", "notype_dense": "pt_notype_dense_",
                     "type": "pt_", "notype": "pt_notype_"},
    },
}

# Default to KuaiRec for backward compatibility
VARIANTS = DATASETS["kuairec"]["variants"]
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


def split_dir_name(dirname, variant_keys=None):
    """Split a save_<prefix>_<rest> dir name into (family, config, variant_key).

    Recognized:
      save_duorec_<variant>
      save_pt_fixrt_<config>_<variant>           (KuaiRec, type)
      save_pt_notype_fixrt_<config>_<variant>    (KuaiRec, notype)
      save_pt_dense_<config>_<dataset>           (new DS, dense, type)
      save_pt_notype_dense_<config>_<dataset>    (new DS, dense, notype)
      save_pt_<config>_<dataset>                 (new DS, non-dense, type)
      save_pt_notype_<config>_<dataset>          (new DS, non-dense, notype)
      save_pt_gru_<config>_<variant>             (GRU-TRIER)
    Returns None if unrecognized.
    """
    if variant_keys is None:
        variant_keys = VARIANT_KEYS
    name = dirname[len("save_"):] if dirname.startswith("save_") else dirname

    if name.startswith("duorec_"):
        rest = name[len("duorec_"):]
        if rest in variant_keys:
            return ("duorec", "duorec", rest)

    # Order matters: check longer prefixes first
    for prefix, family in (("pt_notype_dense_", "trier_notype"),
                           ("pt_dense_", "trier_type"),
                           ("pt_notype_fixrt_", "trier_notype"),
                           ("pt_fixrt_", "trier_type"),
                           ("pt_notype_", "trier_notype"),
                           ("pt_gru_", "gru_trier")):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            for vk in variant_keys:
                if rest.endswith("_" + vk):
                    config = rest[: -(len(vk) + 1)]
                    return (family, config, vk)
    return None


def collect_all_results(proto_filter=None, variant_keys=None):
    """Return list of dicts: {family, config, variant, proto, label, data}."""
    if variant_keys is None:
        variant_keys = VARIANT_KEYS
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
        elif basename.startswith("test_result_greedy_small_bw"):
            # beam-sweep small: test_result_greedy_small_bw10_k10.txt etc.
            proto = "small"
            fname_infer = basename.replace("test_result_greedy_small_", "").replace(".txt", "")
        elif basename.startswith("test_result_greedy_bw"):
            # beam-sweep big: test_result_greedy_bw10_k10.txt etc.
            proto = "big"
            fname_infer = basename.replace("test_result_greedy_", "").replace(".txt", "")
        else:
            continue  # ignore test_result_500.txt etc.
        if proto_filter and proto != proto_filter:
            continue
        parsed = split_dir_name(dirname, variant_keys)
        if parsed is None:
            continue
        family, config, variant = parsed
        # inference mode: explicit from filename, else duorec is topk, TRIER greedy
        # beam-sweep files carry their beam label (e.g. "bw10_k10") as infer
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
        if variant not in variant_keys:
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

def tex_model_label(fam, cfg, infer):
    """Compact paper-style model label for LaTeX table columns."""
    if fam == "baseline":
        base = {"sasrec": "SASRec", "gru4rec": "GRU4Rec",
                "bert4rec": "BERT4Rec"}.get(cfg, cfg)
    elif fam == "duorec":
        base = "DuoRec"
    elif fam == "gru_trier":
        base = "GRU-TRIER"
    else:
        base = "TRIER" if fam == "trier_type" else r"TRIER$_{-t}$"
    # Append short config tag only for TRIER variants (baselines/DuoRec are single config)
    if fam in ("trier_type", "trier_notype", "gru_trier"):
        short = CONFIG_LABELS.get(cfg, cfg)
        # drop the lambda part for column headers to keep them short
        base += " " + short
    # Append infer tag only when both exist (some models have only one mode)
    if fam in ("trier_type", "trier_notype", "gru_trier") and infer and infer != "topk":
        base += " (" + infer + ")"
    return base

def write_latex(rows, path, ds_label="", suffix=""):
    # Layout: rows = (Dataset group × Metric block × Metric), columns = models.
    # Mirrors the TRIER paper's main-results table (Shi et al., TOIS 2024).
    #
    # Metric blocks per dataset:
    #   Acc. : HR@5/10/20 (= Recall) and ND@5/10/20 (= NDCG)   [higher = better]
    #   Div. : ILD@5/10/20 and CC@5/10/20                      [higher = better]
    #   Sim. : CS@5/10/20 (consecutive similarity — EXTRA ROWS) [lower = better]
    #
    # Bolding rule: best (max/min) value per (dataset, metric) is \textbf,
    #               second-best is \underline.
    ACC_METRICS = [
        ("HR@5",  "recall@5"),
        ("HR@10", "recall@10"),
        ("HR@20", "recall@20"),
        ("ND@5",  "ndcg@5"),
        ("ND@10", "ndcg@10"),
        ("ND@20", "ndcg@20"),
    ]
    DIV_METRICS = [
        ("ILD@5",  "ILD@5"),
        ("ILD@10", "ILD@10"),
        ("ILD@20", "ILD@20"),
        ("CC@5",   "CC@5"),
        ("CC@10",  "CC@10"),
        ("CC@20",  "CC@20"),
    ]
    # Lower is better for consecutive similarity.
    SIM_METRICS = [
        ("CS@5",   "CS@5"),
        ("CS@10",  "CS@10"),
        ("CS@20",  "CS@20"),
    ]
    # Direction: True = higher is better, False = lower is better.
    BEST_HIGHER = True

    def direction(metric_name):
        if metric_name.startswith("CS@"):
            return False  # lower similarity = more diverse
        return BEST_HIGHER

    def bold_underline(vals_with_key, metric_name):
        """Return dict keyed by original key -> (formatted_str, is_bold, is_underline)."""
        d = direction(metric_name)
        items = [(k, v) for k, v in vals_with_key.items() if v is not None]
        sorted_items = sorted(items, key=lambda x: x[1], reverse=d)
        out = {}
        if len(sorted_items) >= 1:
            out[sorted_items[0][0]] = True, False
        if len(sorted_items) >= 2:
            out[sorted_items[1][0]] = False, True
        # everything else gets (False, False)
        for k in vals_with_key:
            out.setdefault(k, (False, False))
        return out

    proto_caption = {"big": "full-catalog ranking",
                     "small": "small-matrix protocol"}

    lines = [r"% Auto-generated by analyze_results.py — paper format",
             r"% Requires packages: booktabs, multirow, graphicx, amsmath",
             r"% Bold = best result, underline = second-best (per metric).",
             ""]

    for proto in ("big", "small"):
        proto_rows = [r for r in rows if r["proto"] == proto]
        if not proto_rows:
            continue

        # ---- Build model columns (ordered list of unique model keys) ----
        model_keys = []
        model_order_seen = set()
        for r in sorted(proto_rows, key=sort_key):
            key = (r["family"], r["config"], r["infer"])
            if key not in model_order_seen:
                model_order_seen.add(key)
                model_keys.append(key)

        # ---- Collect per-variant (dataset) data:  model_key -> data_dict ----
        #   variant_data[variant][model_key] = data_dict
        variant_data = {vkey: {} for vkey, _ in VARIANTS}
        for r in proto_rows:
            vkey = r["variant"]
            if vkey not in variant_data:
                continue
            key = (r["family"], r["config"], r["infer"])
            variant_data[vkey][key] = r["data"]

        # ---- Best/underline per (variant, metric) across model columns ----
        #   highlight[(variant_key, metric_slug)] = {model_key: (bold, underline)}
        highlight = {}
        for vkey in variant_data:
            all_metrics = ACC_METRICS + DIV_METRICS + SIM_METRICS
            for _, metric_slug in all_metrics:
                inner = variant_data[vkey]  # {model_key: data_dict}
                vals = {mk: get_metric(mdata, metric_slug)
                        for mk, mdata in inner.items()}
                highlight[(vkey, metric_slug)] = bold_underline(vals, metric_slug)

        # ---- LaTeX assembly ----
        cap_ds = tex_escape(ds_label) if ds_label else "all datasets"
        # Wide (>=2 dataset variants): spans both columns; narrow (single-dataset):
        # fits one ACM column at footnotesize without resizebox stretch.
        wide = len(VARIANTS) >= 2
        env = "table*" if wide else "table"
        lines.append(r"\begin{" + env + r"}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Overall comparison on " + cap_ds + r" ("
                     + proto_caption.get(proto, proto)
                     + r"). Best results are in \textbf{bold}, second-best "
                       r"in \underline{underline}. CS@$K$ = average "
                       r"consecutive item similarity (lower is more diverse).}")
        lines.append(r"\label{tab:results" + suffix + "_" + proto + r"}")
        if wide:
            lines.append(r"\resizebox{\textwidth}{!}{%")
        else:
            lines.append(r"\setlength{\tabcolsep}{4pt}")
            lines.append(r"\footnotesize")

        # Column spec: 3 label cols (dataset group + sub-block + metric) + N model cols
        n_models = len(model_keys)
        lines.append(r"\begin{tabular}{lllc" + "c" * n_models + r"}")
        lines.append(r"\toprule")

        # Header: three label cols (blank) + one header per model column
        hdr = r"& & & " + " & ".join(tex_model_label(f, c, i)
                                     for f, c, i in model_keys) + r" \\"
        lines.append(hdr)
        lines.append(r"\midrule")

        for v_idx, (vkey, vlabel) in enumerate(VARIANTS):
            if not variant_data[vkey]:
                continue
            ds_label_short = vlabel.replace("kuairec_", "").replace("_", "-")
            ds_label_short = ds_label_short.replace("KUAIREC", "KuaiRec")
            # Metric blocks in order: Acc., Div., Sim.
            blocks = [
                ("Acc.", ACC_METRICS),
                ("Div.", DIV_METRICS),
                ("Sim.", SIM_METRICS),
            ]
            first_line = True
            for block_label, metrics in blocks:
                for m_idx, (m_label, m_slug) in enumerate(metrics):
                    # Dataset name on its first line
                    if first_line:
                        ds_cell = r"\multirow{" + str(sum(len(ms) for _, ms in blocks)) + r"}{*}{" + ds_label_short + r"}"
                        first_line = False
                    else:
                        ds_cell = ""
                    # Sub-block label on its first line of that block
                    is_block_first = (m_idx == 0)
                    if is_block_first:
                        block_cell = r"\multirow{" + str(len(metrics)) + r"}{*}{" + block_label + r"}"
                    else:
                        block_cell = ""
                    # Metric cells across models
                    cells = []
                    for mkey in model_keys:
                        data = variant_data[vkey].get(mkey)
                        v = get_metric(data, m_slug)
                        if v is None:
                            cells.append("--")
                            continue
                        cell_str = f"{v:.4f}"
                        bold, ul = highlight[(vkey, m_slug)].get(mkey, (False, False))
                        if bold:
                            cell_str = r"\textbf{" + cell_str + r"}"
                        elif ul:
                            cell_str = r"\underline{" + cell_str + r"}"
                        cells.append(cell_str)
                    # Row: dataset_label & block_label & metric_name & cell1 & cell2 & ... \\
                    lines.append(ds_cell + " & " + block_cell + " & " + m_label + " & " + " & ".join(cells) + r" \\")
                # add small gap between blocks (but not after the last block)
                # use \addlinespace from booktabs
            # add extra spacing between dataset groups (not after the last one)
            if v_idx < len(VARIANTS) - 1:
                lines.append(r"\addlinespace[0.3em]")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if wide:
            lines.append(r"}")
        lines.append(r"\end{" + env + r"}")
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
    parser.add_argument("--dataset", default="kuairec",
                        choices=list(DATASETS.keys()) + ["all"],
                        help="Which dataset to analyze (default: kuairec)")
    args = parser.parse_args()

    # Collect variant keys for the selected dataset(s)
    if args.dataset == "all":
        variant_keys = []
        for ds in DATASETS.values():
            variant_keys.extend(v[0] for v in ds["variants"])
        ds_label = "All datasets"
    else:
        ds_info = DATASETS[args.dataset]
        variant_keys = [v[0] for v in ds_info["variants"]]
        ds_label = ds_info["label"]

    print(f"Collecting results for {ds_label}...")
    rows = collect_all_results(proto_filter=args.proto, variant_keys=variant_keys)
    print(f"Found {len(rows)} result files\n")

    if not rows:
        print(f"No results found for {ds_label}. Run eval scripts first.")
        return

    # Update VARIANTS for display (LaTeX table headers, sort order)
    global VARIANTS, VARIANT_KEYS
    if args.dataset == "all":
        VARIANTS = [(vk, vk.replace("kuairec_", "").replace("_", "-"))
                    for ds in DATASETS.values() for vk, _ in ds["variants"]]
    else:
        VARIANTS = DATASETS[args.dataset]["variants"]
    VARIANT_KEYS = [v[0] for v in VARIANTS]

    protos = [args.proto] if args.proto else ["big", "small"]
    for p in protos:
        print_console_table(rows, p)

    suffix = f"_{args.dataset}" if args.dataset != "kuairec" else ""
    write_csv(rows, os.path.join(SCRIPT_DIR, f"results_summary{suffix}.csv"))
    write_latex(rows, os.path.join(SCRIPT_DIR, f"results_tables{suffix}.tex"),
                ds_label=ds_label, suffix=suffix)


if __name__ == "__main__":
    main()
