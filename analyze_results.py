#!/usr/bin/env python3
"""
Analyze results from all TRIER variants and baselines.
Reads test_result.txt and valid_result.txt from each save directory.
Generates a comparison table.
"""
import os
import sys
import glob
import json
import ast

VARIANTS = [
    "kuairec_highest_individual",
    "kuairec_highest_average",
    "kuairec_first_individual",
    "kuairec_first_average",
]

# Model directories to check: (label, path_pattern)
MODEL_DIRS = [
    ("RT",              "./save_rt_{var}"),
    ("PT (lamb=0.1)",   "./save_pt_{var}"),
    ("PT (no consec)",  "./save_pt_no_consec_{var}"),
    ("PT (lamb=0.01)",  "./save_pt_lamb001_{var}"),
    ("PT (lamb=0)",     "./save_pt_lamb0_{var}"),
]

# Baseline results (if available)
BASELINE_DIRS = [
    ("GRU4Rec",  "./baseline_results_kuairec_first_average/gru4rec_results.txt"),
    ("SASRec",   "./baseline_results_kuairec_first_average/sasrec_results.txt"),
    ("BERT4Rec", "./baseline_results_kuairec_first_average/bert4rec_results.txt"),
]


def parse_result_file(filepath):
    """Parse a result txt file into a dict of metrics.
    Result files contain one Python dict literal per line (one per epoch).
    We take the last (most recent) line."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if not lines:
            return None
        # Try each line from bottom up — last valid dict is the latest epoch
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            # Skip log/info lines
            if line.startswith('Loading') or line.startswith('Warning') or line.startswith('Using') or line.startswith('Best'):
                continue
            # Try Python dict literal (main format from main_pt.py / main_rt.py)
            try:
                d = ast.literal_eval(line)
                if isinstance(d, dict):
                    return d
            except:
                pass
            # Try JSON
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    return d
            except:
                pass
        # Fallback: try line-by-line key: value format (baselines)
        result = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('epoch') or line.startswith('Loading') or line.startswith('Warning') or line.startswith('Using'):
                continue
            for sep in [': ', '= ']:
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val_str = parts[1].strip()
                        try:
                            val = float(val_str)
                            result[key] = val
                        except ValueError:
                            pass
                        break
        return result if result else None
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return None


def metric_val(d, key):
    """Get metric value, trying with and without _f suffix."""
    if d is None:
        return 0.0
    if key in d:
        val = d[key]
        return float(val) if val is not None else 0.0
    suffixed = f"{key}_f"
    if suffixed in d:
        val = d[suffixed]
        return float(val) if val is not None else 0.0
    return 0.0


def get_latest_epoch(save_dir):
    """Find the latest checkpoint epoch number."""
    model_dir = os.path.join(save_dir, "model")
    if not os.path.exists(model_dir):
        return "?"
    ckpts = glob.glob(os.path.join(model_dir, "duorec-*.pth"))
    if not ckpts:
        return "?"
    epochs = []
    for f in ckpts:
        try:
            ep = int(f.split("duorec-")[1].split(".pth")[0])
            epochs.append(ep)
        except:
            pass
    return str(max(epochs)) if epochs else "?"


def main():
    print("=" * 120)
    print("COMPARISON SUMMARY")
    print("=" * 120)

    # Header
    header = f"{'Variant':<20} | {'Model':<18} | {'Best Ep':<7} | {'V_R@5':<8} {'V_R@10':<8} {'V_R@20':<8} {'V_N@10':<8} {'V_N@20':<8} {'V_M@10':<8} {'V_ILD@10':<9} {'V_CS@10':<8} {'V_CC@10':<8} | {'T_R@10':<8} {'T_N@10':<8} {'T_ILD@10':<9} {'T_CS@10':<8} {'T_CC@10':<8}"
    print(header)
    print("-" * 120)

    for var in VARIANTS:
        for label, path_pattern in MODEL_DIRS:
            save_dir = path_pattern.replace("{var}", var)
            if not os.path.exists(save_dir):
                continue

            # Parse test and valid results
            test_file = os.path.join(save_dir, "test_result.txt")
            valid_file = os.path.join(save_dir, "valid_result.txt")
            test_data = parse_result_file(test_file)
            valid_data = parse_result_file(valid_file)

            if test_data is None and valid_data is None:
                continue

            best_ep = get_latest_epoch(save_dir)

            # Validation metrics
            v_r5  = metric_val(valid_data, "recall@5")
            v_r10 = metric_val(valid_data, "recall@10")
            v_r20 = metric_val(valid_data, "recall@20")
            v_n10 = metric_val(valid_data, "ndcg@10")
            v_n20 = metric_val(valid_data, "ndcg@20")
            v_m10 = metric_val(valid_data, "mrr@10")
            v_ild10 = metric_val(valid_data, "ILD@10")
            v_cs10  = metric_val(valid_data, "CS@10")
            v_cc10  = metric_val(valid_data, "CC@10")

            # Test metrics
            t_r10 = metric_val(test_data, "recall@10")
            t_n10 = metric_val(test_data, "ndcg@10")
            t_ild10 = metric_val(test_data, "ILD@10")
            t_cs10  = metric_val(test_data, "CS@10")
            t_cc10  = metric_val(test_data, "CC@10")

            row = f"{var:<20} | {label:<18} | {best_ep:<7} | {v_r5:<8.4f} {v_r10:<8.4f} {v_r20:<8.4f} {v_n10:<8.4f} {v_n20:<8.4f} {v_m10:<8.4f} {v_ild10:<9.4f} {v_cs10:<8.4f} {v_cc10:<8.4f} | {t_r10:<8.4f} {t_n10:<8.4f} {t_ild10:<9.4f} {t_cs10:<8.4f} {t_cc10:<8.4f}"
            print(row)

    # Baselines (only first_average)
    print("-" * 120)
    print("Baselines (first_average):")
    print("-" * 120)
    for label, path in BASELINE_DIRS:
        if not os.path.exists(path):
            continue
        data = parse_result_file(path)
        if data is None:
            continue
        r5  = metric_val(data, "Recall@5")
        r10 = metric_val(data, "Recall@10")
        r20 = metric_val(data, "Recall@20")
        n5  = metric_val(data, "NDCG@5")
        n10 = metric_val(data, "NDCG@10")
        n20 = metric_val(data, "NDCG@20")
        m5  = metric_val(data, "MRR@5")
        m10 = metric_val(data, "MRR@10")
        m20 = metric_val(data, "MRR@20")
        print(f"{'first_average':<20} | {label:<18} | {'500':<7} | {r5:<8.4f} {r10:<8.4f} {r20:<8.4f} {n10:<8.4f} {n20:<8.4f} {m10:<8.4f} {'N/A':<9} {'N/A':<8} {'N/A':<8} | {r10:<8.4f} {n10:<8.4f} {'N/A':<9} {'N/A':<8} {'N/A':<8}")

    print("=" * 120)
    print("Analysis complete.")
    print("=" * 120)


if __name__ == '__main__':
    main()
