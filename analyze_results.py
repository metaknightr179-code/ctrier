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
    python3 analyze_results.py --dataset kuairec --dense_only  # dense KuaiRec only
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
FAMILY_ORDER = {"baseline": 0, "duorec": 1, "trier_notype": 2, "trier_author": 3,
                "trier_type": 4, "trier_typeauthor": 5, "trier_typemusic": 6,
                "trier_typedur": 7, "gru_trier": 8}

# config suffix -> display label (lambda sweep)
CONFIG_LABELS = {
    "nodiv": "No-Div",
    "lamb0002": "$\\lambda$=0.002",
    "lamb0005": "$\\lambda$=0.005",
    "lamb0005_consec0001": "$\\lambda$=0.005+C0.001",
    "lamb0005_consec005": "$\\lambda$=0.005+C0.05",
    "lamb0005_consec01": "$\\lambda$=0.005+C0.1",
    "lamb001": "$\\lambda$=0.01",
    "lamb005": "$\\lambda$=0.05",
    "lamb01": "$\\lambda$=0.1",
}
CONFIG_ORDER = ["nodiv", "lamb0002", "lamb0005", "lamb0005_consec0001",
                "lamb0005_consec005", "lamb0005_consec01",
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
            m = re.match(r'\s*(Recall|MRR|NDCG|ILD|CS|CC|MaxRun)@(\d+):\s*([-\d.]+)', line)
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
                           ("pt_typeauthor_fixrt_", "trier_typeauthor"),
                           ("pt_typemusic_fixrt_", "trier_typemusic"),
                           ("pt_typedur_fixrt_", "trier_typedur"),
                           ("pt_author_fixrt_", "trier_author"),
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


# PT directory prefixes whose models were trained with -dense.
# NOTE: the side-info families (author/music/dur) keep the "_fixrt_" name from
# the shared-RT pipeline but are ALL trained dense — they must be included by
# --dense_only even though their dirname does not contain the substring "dense".
DENSE_DIR_PREFIXES = (
    "save_pt_dense_",
    "save_pt_notype_dense_",
    "save_pt_typeauthor_fixrt_",
    "save_pt_typemusic_fixrt_",
    "save_pt_typedur_fixrt_",
    "save_pt_author_fixrt_",
    "save_pt_music_fixrt_",
    "save_pt_authormusic_fixrt_",
    "save_pt_typeall_fixrt_",
)


def is_dense_dir(dirname):
    return dirname.startswith(DENSE_DIR_PREFIXES)


def collect_all_results(proto_filter=None, variant_keys=None, dense_only=False):
    """Return list of dicts: {family, config, variant, proto, label, data}.

    If dense_only=True, only include dirs trained with -dense
    (see DENSE_DIR_PREFIXES).
    """
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
        if dense_only and not is_dense_dir(dirname):
            continue
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
    "MaxRun@5": ["MaxRun@5", "maxrun@5"],
    "MaxRun@10": ["MaxRun@10", "maxrun@10"],
    "MaxRun@20": ["MaxRun@20", "maxrun@20"],
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
    fam_prefix = {"trier_type": "TRIER(type) ", "trier_notype": "TRIER(notype) ",
                  "trier_author": "TRIER(author) ",
                  "trier_music": "TRIER(music) ",
                  "trier_authormusic": "TRIER(author+music) ",
                  "trier_typeauthor": "TRIER(type+author) ",
                  "trier_typemusic": "TRIER(type+music) ",
                  "trier_typeall": "TRIER(type+author+music) ",
                  "trier_typedur": "TRIER(type+dur) "}.get(fam, "TRIER(notype) ")
    return fam_prefix + CONFIG_LABELS.get(cfg, cfg)


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
                    "CC@5", "CC@10", "CC@20",
                    "MaxRun@5", "MaxRun@10", "MaxRun@20"]
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
        # Type-based families (type on, possibly + one side channel) keep plain TRIER;
        # type-off families (notype, author-only) get the -t subscript.
        type_on = fam in ("trier_type", "trier_typeauthor", "trier_typemusic", "trier_typedur")
        base = "TRIER" if type_on else r"TRIER$_{-t}$"
        # Side-channel superscript tag
        side_tag = {"trier_typeauthor": r"$^{+a}$",
                    "trier_typemusic": r"$^{+m}$",
                    "trier_typedur": r"$^{+d}$"}.get(fam, "")
        base += side_tag
    # Append short config tag only for TRIER variants (baselines/DuoRec are single config)
    trier_fams = ("trier_type", "trier_notype", "trier_author", "trier_typeauthor",
                  "trier_typemusic", "trier_typedur", "gru_trier")
    if fam in trier_fams:
        short = CONFIG_LABELS.get(cfg, cfg)
        # drop the lambda part for column headers to keep them short
        base += " " + short
    # Append infer tag only when both exist (some models have only one mode)
    if fam in trier_fams and infer and infer != "topk":
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
    #   Run. : MaxRun@5/10/20 (longest similar-item run)        [lower = better]
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
    # Lower is better for MaxRun (longest contiguous run of similar items).
    RUN_METRICS = [
        ("MR@5",   "MaxRun@5"),
        ("MR@10",  "MaxRun@10"),
        ("MR@20",  "MaxRun@20"),
    ]
    # Direction: True = higher is better, False = lower is better.
    BEST_HIGHER = True

    def direction(metric_name):
        if metric_name.startswith("CS@"):
            return False  # lower similarity = more diverse
        if metric_name.startswith("MaxRun@") or metric_name.startswith("MR@"):
            return False  # shorter run = less repetition
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
            all_metrics = ACC_METRICS + DIV_METRICS + SIM_METRICS + RUN_METRICS
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
                       r"consecutive item similarity (lower is more diverse); "
                       r"MR@$K$ = MaxRun, longest contiguous run of mutually "
                       r"similar items (lower is less repetitive).}")
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
            # Metric blocks in order: Acc., Div., Sim., Run.
            blocks = [
                ("Acc.", ACC_METRICS),
                ("Div.", DIV_METRICS),
                ("Sim.", SIM_METRICS),
                ("Run.", RUN_METRICS),
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
# Embedding ablation table (2^3 CONTENT x AUTHOR x MUSIC at λ=0.01)
# =============================================================================
def write_embedding_ablation_tex(variant="kuairec_first_average", lamb=0.01,
                                 proto="big", path=None):
    """Generate the 8-cell embedding-ablation LaTeX table.

    Reads result files produced by eval_embedding_ablation_kuairec.sh, which
    writes results back into the canonical checkpoint directories.

    CELL LAYOUT (2^3 = 8 combinations of three content flags):

      Model        Type  Author  Music  Description
      ----------------------------------------------------------
      notype        —       —       —   backbone, no side info at all
      type          ✓       —       —   + RecFormer-type content embeddings
      author        —       ✓       —   notype + author embeddings
      music         —       —       ✓   notype + music embeddings
      authormusic   —       ✓       ✓   notype + both side channels
      typeauthor    ✓       ✓       —   type + author
      typemusic     ✓       —       ✓   type + music
      typeall       ✓       ✓       ✓   type + author + music (full PACER)

    DIRECTORY MAPPING — IMPORTANT naming-trap notes:
      * notype / type baselines (dense sweep at λ=0.01):
            save_pt_notype_dense_lamb001_<variant>   ← trained -lamb 0.01
            save_pt_dense_lamb001_<variant>          ← trained -lamb 0.01
        (The legacy suffix lamb01 means trained -lamb 0.1 — AVOID here.)

      * 6 side-info families (trained by train_pt_sideinfo01_fixrt.sh,
        which despite its lamb01 suffix actually uses -lamb 0.01):
            save_pt_author_fixrt_lamb01_<variant>       ... etc.

      * eval_embedding_ablation_kuairec.sh already copies results BACK into
        these canonical checkpoint dirs: test_result.txt (greedy),
        test_result_small.txt (greedy small), test_result_topk.txt / _topk_small.

    METRIC COLS (same as six-cell): HR/NDCG@{5,10,20}, ILD@20, CC@20, CS@20, MaxRun@20
    """
    # ── Directory mapping ──────────────────────────────────────────────
    # Baselines: use lamb001 = genuinely trained at λ=0.01
    # Sideinfo:  use fixrt_lamb01 — also trained at λ=0.01 (see train script)
    cells = [
        # (label,          dir_prefix,                                       type_on, author_on, music_on, question)
        ("notype",         "save_pt_notype_dense_lamb001",                False, False, False, "no side info"),
        ("type",           "save_pt_dense_lamb001",                        True,  False, False, "+ content (RecFormer)"),
        ("author",         "save_pt_author_fixrt_lamb01",                  False, True,  False, "+ author embeddings"),
        ("music",          "save_pt_music_fixrt_lamb01",                  False, False, True,  "+ music embeddings"),
        ("authormusic",    "save_pt_authormusic_fixrt_lamb01",            False, True,  True,  "+ both side channels"),
        ("typeauthor",     "save_pt_typeauthor_fixrt_lamb01",              True,  True,  False, "+ content + author"),
        ("typemusic",      "save_pt_typemusic_fixrt_lamb01",               True,  False, True,  "+ content + music"),
        ("typeall",        "save_pt_typeall_fixrt_lamb01",                 True,  True,  True,  "+ content + both (full PACER)"),
    ]

    metric_cols = [
        ("HR@5",   "recall@5_f"),  ("HR@10",  "recall@10_f"),  ("HR@20",  "recall@20_f"),
        ("NDCG@5", "ndcg@5_f"),    ("NDCG@10", "ndcg@10_f"),    ("NDCG@20", "ndcg@20_f"),
        ("ILD@20", "ILD@20"),  ("CC@20", "CC@20"),
        ("CS@20",  "CS@20"),   ("MaxRun@20", "MaxRun@20"),
    ]

    suffix = "_small" if proto == "small" else ""
    fname = f"test_result{suffix}.txt"

    rows_data, missing = [], []
    for label, prefix, has_type, has_author, has_music, qn in cells:
        fpath = os.path.join(SCRIPT_DIR, f"{prefix}_{variant}", fname)
        data = parse_dict_result(fpath)
        if data is None:
            missing.append(fpath)
        rows_data.append((label, has_type, has_author, has_music, qn, data))

    if missing:
        print(f"\n⚠  Embedding ablation: MISSING {len(missing)} result files. "
              f"Run CUDA_VISIBLE_DEVICES=0 bash eval_embedding_ablation_kuairec.sh. "
              f"\n   Example missing: {missing[0]}")

    def yn(b): return "\\checkmark" if b else "—"

    def fmt(v):
        return "\\textemdash" if v is None else f"{v:.4f}"

    out_lines = [
        r"% 8-cell embedding ablation (2^3 = Content(RecFormer-type) × Author × Music)",
        r"% Generated by analyze_results.py --embedding_ablation",
        r"% Variant: " + variant + r"   λ=" + str(lamb) + r"   proto=" + proto,
        r"% All checkpoints are trained at -lamb 0.01 (DIR NAME suffix trap: lamb01/lamb001",
        r"% both mean λ=0.01 here — see train_pt_sideinfo01_fixrt.sh and train_dense_fixrt.sh).",
        r"% CS@20 and MaxRun@20 are lower-is-better.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Eight-cell embedding ablation on \emph{" +
            variant.replace("kuairec_","").replace("_","-") +
            r"} (KuaiRec, $\lambda=" + str(lamb) +
            r"$, " + proto + r"-matrix protocol). The backbone (TRIER-PT, no RT augmentation here) "
            r"is \emph{notype}: only ID embeddings. \textbf{type} adds the RecFormer-style learnable category embeddings, "
            r"\textbf{author} the per-item creator embeddings, and \textbf{music} the per-item audio embeddings; "
            r"full PACER combines all three. Best HR/NDCG/ILD/CC values per block are \textbf{bold}; "
            r"for CS@20 and MaxRun@20 lower is better so their best is also bold.}",
        r"\label{tab:embedding_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc " + "r"*len(metric_cols) + r"}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & "
        r"\multicolumn{1}{c}{\textbf{Type}} & "
        r"\multicolumn{1}{c}{\textbf{Author}} & "
        r"\multicolumn{1}{c}{\textbf{Music}} & "
        r"\multirow{2}{*}{\textbf{Config}} & "
        + " & ".join(r"\multicolumn{1}{c}{\textbf{" + mc[0] + r"}}" for mc in metric_cols) + r" \\",
        r" & & & & & " + " & ".join(r"\scriptsize " + mc[0] for mc in metric_cols) + r" \\",
        r"\midrule",
    ]
    for label, has_type, has_author, has_music, qn, data in rows_data:
        vals = " & ".join(fmt(get_metric(data, m)) for _, m in metric_cols)
        out_lines.append(
            f"{label} & {yn(has_type)} & {yn(has_author)} & {yn(has_music)} & {qn} & {vals} \\\\"
        )
    out_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]

    table_str = "\n".join(out_lines)

    if path is None:
        path = os.path.join(SCRIPT_DIR,
            f"embedding_ablation_{variant.replace('kuairec_','')}_{proto}.tex")
    with open(path, "w") as f:
        f.write(table_str + "\n")
    print(f"Embedding ablation table written: {path}")

    # Console summary
    print("\n" + "=" * 110)
    print(f"EMBEDDING ABLATION (variant={variant}, λ={lamb}, proto={proto})")
    print("=" * 110)
    header = (f"{'Model':<14} {'T':>2} {'A':>2} {'M':>2}  "
              + " ".join(f"{mc[0]:>10}" for mc in metric_cols))
    print(header)
    print("-" * len(header))
    for label, has_type, has_author, has_music, qn, data in rows_data:
        vals = " ".join(
            f"{get_metric(data, m):>10.4f}" if get_metric(data, m) is not None
            else f"{'MISS':>10}" for _, m in metric_cols)
        mark = lambda b: "✓" if b else "—"
        print(f"{label:<14} {mark(has_type):>2} {mark(has_author):>2} "
              f"{mark(has_music):>2}  {vals}")
    print("=" * 110)

    return table_str


# =============================================================================
# Six-cell component ablation table (Content x Order-loss x Order-score)
# =============================================================================
def write_sixcell_tex(outdir, variant="kuairec_first_average", lamb=0.01,
                      proto="big", path=None):
    """Generate the six-cell ablation LaTeX table consumed by approach.tex.

    Reuses result files produced by eval_sixcell_ablation_kuairec.sh.
    Cell ordering and ablation layout match the figure in the issue comment:

      Model       Content   Order loss(=γ_o L_order)   Order score(=-λ_c C_s(j))   Question
      TRIER       否        否                          否                         original backbone
      TRIER-C     是        否                          否                         does content help?
      TRIER-L     否        是                          否                         training consec loss alone
      TRIER-S     否        否                          是                         inference penalty alone
      PACER-LS    否        是                          是                         train/infer order mechanisms together
      PACER Full  是        是                          是                         complete model

    Order loss = the soft consec training loss γ_o L_order (not L_div!),
    Order score = the hard inference-time penalty -λ_c C_s(j).
    """
    # Cell definition: label, dir_name, Content, OrderLoss, OrderScore, question
    # We map to the directory structure used by eval_sixcell_ablation_kuairec.sh.
    cells = [
        ("TRIER",      "TRIER",      False, False, False, "原始 backbone"),
        ("TRIER-C",    "TRIER-C",    True,  False, False, "内容表示是否提高准确率"),
        ("TRIER-L",    "TRIER-L",    False, True,  False, "可微顺序损失单独作用"),
        ("TRIER-S",    "TRIER-S",    False, False, True,  "推理惩罚单独作用"),
        ("PACER-LS",   "PACER-LS",   False, True,  True,  "训练与推理顺序机制是否互补"),
        ("PACER Full", "PACER-Full", True,  True,  True,  "完整模型是否取得最好权衡"),
    ]

    # Which metrics to include and at which k
    metric_cols = [
        # Accuracy (HR/NDCG)
        ("HR@5",   "recall@5_f"),
        ("HR@10",  "recall@10_f"),
        ("HR@20",  "recall@20_f"),
        ("NDCG@5", "ndcg@5_f"),
        ("NDCG@10","ndcg@10_f"),
        ("NDCG@20","ndcg@20_f"),
        # Diversity
        ("ILD@20", "ILD@20"),
        ("CC@20",  "CC@20"),
        # Repetition
        ("CS@20",  "CS@20"),
        ("MaxRun@20", "MaxRun@20"),
    ]

    suffix = "_small" if proto == "small" else ""

    rows_data = []
    missing = []
    for label, dirname, has_content, has_orderloss, has_orderscore, qn in cells:
        fpath = os.path.join(outdir, dirname, f"test_result{suffix}.txt")
        data = parse_dict_result(fpath)
        if data is None:
            missing.append(fpath)
            rows_data.append((label, has_content, has_orderloss, has_orderscore, qn, None))
        else:
            rows_data.append((label, has_content, has_orderloss, has_orderscore, qn, data))

    if missing:
        print(f"\n⚠  Six-cell ablation: MISSING {len(missing)} result files — run "
              f"CUDA_VISIBLE_DEVICES=0 bash eval_sixcell_ablation_kuairec.sh first.\n"
              f"   Missing examples: {missing[0]}")

    def yn(b): return "\\checkmark" if b else "—"

    def fmt(v):
        if v is None: return "\\textemdash"
        return f"{v:.4f}"

    # Build LaTeX
    out_lines = [
        r"% Six-cell component ablation (Content × Order loss × Order score)",
        r"% Generated by analyze_results.py --sixcell",
        r"% Variant: " + variant + r"   λ=" + str(lamb) + r"   proto=" + proto,
        r"% Order loss = \gamma_o L_order (soft consec training loss, NOT L_div)",
        r"% Order score = -\lambda_c C_s(j) (hard inference-time penalty)",
        r"% MaxRun lower = better (1 = no adjacent similarities, k = fully repetitive)",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Six-cell component ablation on \emph{" + variant.replace("kuairec_","").replace("_","-") +
                   r"} (KuaiRec, $\lambda=" + str(lamb) + r"$, " + proto + r"-matrix protocol). "
                   r"\emph{Order loss} denotes the training-time soft consecutive loss $\gamma_o \mathcal{L}_{order}$ "
                   r"(distinct from the TRIER ILD diversity loss $\mathcal{L}_{div}$), and \emph{Order score} denotes "
                   r"the inference-time hard adjacent penalty $-\lambda_c C_s(j)$. Best in each accuracy/diversity block "
                   r"is \textbf{bold}; for CS@20 and MaxRun@20 \emph{lower is better}, so their best is also bold.}",
        r"\label{tab:sixcell_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllll " + "r"*len(metric_cols) + r"}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & "
        r"\multirow{2}{*}{\textbf{Content}} & "
        r"\multirow{2}{*}{\textbf{Order loss}} & "
        r"\multirow{2}{*}{\textbf{Order score}} & "
        r"\multirow{2}{*}{\textbf{Question}} & "
        + " & ".join(r"\multicolumn{1}{c}{\textbf{" + mc[0] + r"}}" for mc in metric_cols) + r" \\",
        r" & & & & & " + " & ".join(r"\scriptsize " + mc[0] for mc in metric_cols) + r" \\",
        r"\midrule",
    ]
    for label, has_content, has_orderloss, has_orderscore, qn, data in rows_data:
        vals = " & ".join(fmt(get_metric(data, m)) for _, m in metric_cols)
        out_lines.append(
            f"{label} & {yn(has_content)} & {yn(has_orderloss)} & {yn(has_orderscore)} & {qn} & {vals} \\\\"
        )
    out_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]

    table_str = "\n".join(out_lines)

    if path is None:
        path = os.path.join(SCRIPT_DIR, f"sixcell_ablation_{variant.replace('kuairec_','')}.tex")

    with open(path, "w") as f:
        f.write(table_str + "\n")
    print(f"Six-cell ablation table written: {path}")

    # Also print a compact console summary
    print("\n" + "=" * 100)
    print(f"SIX-CELL ABLATION  (variant={variant}, λ={lamb}, proto={proto})")
    print("=" * 100)
    header = f"{'Model':<12} {'C':>3} {'L':>3} {'S':>3} " + " ".join(f"{mc[0]:>10}" for mc in metric_cols)
    print(header)
    print("-" * len(header))
    for label, has_content, has_orderloss, has_orderscore, qn, data in rows_data:
        vals = " ".join(f"{get_metric(data, m):>10.4f}" if get_metric(data, m) is not None else f"{'MISS':>10}"
                        for _, m in metric_cols)
        mark = lambda b: "✓" if b else "—"
        print(f"{label:<12} {mark(has_content):>3} {mark(has_orderloss):>3} {mark(has_orderscore):>3} {vals}")
    print("=" * 100)

    return table_str


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
    parser.add_argument("--dense_only", action="store_true",
                        help="Only show dense checkpoints (save_pt_dense_* / save_pt_notype_dense_*)")
    parser.add_argument("--sixcell", action="store_true",
                        help="Additionally generate the six-cell component ablation table "
                             "(reads from ./sixcell_firstavg/ — run eval_sixcell_ablation_kuairec.sh first)")
    parser.add_argument("--embedding_ablation", action="store_true",
                        help="Additionally generate the 8-cell embedding-ablation table "
                             "(reads canonical checkpoint dirs populated by "
                             "eval_embedding_ablation_kuairec.sh)")
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
    rows = collect_all_results(proto_filter=args.proto, variant_keys=variant_keys,
                               dense_only=args.dense_only)
    print(f"Found {len(rows)} result files\n")

    if not rows and not args.sixcell:
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
        if rows:
            print_console_table(rows, p)

    if rows:
        suffix = f"_{args.dataset}" if args.dataset != "kuairec" else ""
        write_csv(rows, os.path.join(SCRIPT_DIR, f"results_summary{suffix}.csv"))
        write_latex(rows, os.path.join(SCRIPT_DIR, f"results_tables{suffix}.tex"),
                    ds_label=ds_label, suffix=suffix)

    # --- Six-cell component ablation ---
    if args.sixcell:
        for proto in protos:
            write_sixcell_tex(
                outdir=os.path.join(SCRIPT_DIR, "sixcell_firstavg"),
                variant="kuairec_first_average",
                lamb=0.01,
                proto=proto,
            )

    # --- Embedding (2^3) ablation ---
    if args.embedding_ablation:
        for proto in protos:
            write_embedding_ablation_tex(
                variant="kuairec_first_average",
                lamb=0.01,
                proto=proto,
            )


if __name__ == "__main__":
    main()
