#!/usr/bin/env python3
"""
Analyze TRIER RT/PT training results from all Kuairec variants.
Reads train_result.txt, valid_result.txt, test_result.txt from each save directory
and produces a comparative summary.

Usage:
    python3 analyze_results.py [--base_dir .]
"""
import argparse
import os
import re
import ast
from collections import defaultdict


VARIANTS = [
    "kuairec_highest_individual",
    "kuairec_highest_average",
    "kuairec_first_individual",
    "kuairec_first_average",
]

# Directories to analyze: (label, prefix)
DIRS = [
    ("RT", "save_rt_"),
    ("PT (with consec)", "save_pt_"),
    ("PT (no consec)", "save_pt_no_consec_"),
]


def parse_result_file(filepath):
    """Parse a result file where each line is either 'epoch X loss Y' or a dict string."""
    if not os.path.exists(filepath):
        return None
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try to parse as dict (valid/test results)
            if line.startswith('{'):
                try:
                    d = ast.literal_eval(line)
                    results.append(d)
                except:
                    pass
            else:
                # Parse 'epoch X loss Y' format (train results)
                m = re.match(r'epoch\s+(\d+)\s+loss\s+([\d.]+)', line)
                if m:
                    results.append({
                        'epoch': int(m.group(1)),
                        'loss': float(m.group(2)),
                    })
    return results if results else None


def metric_val(d, key, default=0.0):
    """Read a metric from dict, trying both _f-suffixed and unsuffixed keys."""
    if d is None:
        return default
    v = d.get(key)
    if v is None and key.endswith('_f'):
        v = d.get(key[:-2])
    if v is None and not key.endswith('_f'):
        v = d.get(key + '_f')
    return default if v is None else v


def find_best_epoch(valid_results, metric='recall@5_f'):
    """Find the epoch with the best validation metric (tries both _f and unsuffixed keys)."""
    if not valid_results:
        return None, None
    best_val = -1
    best_epoch = None
    for r in valid_results:
        val = metric_val(r, metric, -1)
        if val > best_val:
            best_val = val
            best_epoch = r.get('epoch')
    return best_epoch, best_val


def print_table(headers, rows):
    """Print a formatted table."""
    if not rows:
        return
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    # Print header
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))
    # Print rows
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def analyze_variant(variant, base_dir):
    """Analyze all model types for a given variant."""
    print(f"\n{'='*70}")
    print(f"VARIANT: {variant}")
    print(f"{'='*70}")

    for label, prefix in DIRS:
        save_dir = os.path.join(base_dir, f"{prefix}{variant}")
        if not os.path.exists(save_dir):
            print(f"\n[{label}] Directory not found: {save_dir} — skipping")
            continue

        print(f"\n--- {label} ({prefix}{variant}) ---")

        # Parse train results
        train_results = parse_result_file(os.path.join(save_dir, 'train_result.txt'))
        if train_results:
            epochs_trained = len(train_results)
            last_epoch = train_results[-1].get('epoch', epochs_trained)
            last_loss = train_results[-1].get('loss', None)
            # Find best (lowest) train loss
            valid_losses = [r for r in train_results if r.get('loss') is not None]
            print(f"  Training: {epochs_trained} epochs completed (last epoch {last_epoch})")
            if last_loss is not None:
                print(f"  Final train loss: {last_loss:.4f}")
            else:
                print(f"  Final train loss: N/A (train_result.txt format: {list(train_results[-1].keys())[:5]}...)")
            if valid_losses:
                best_train = min(valid_losses, key=lambda x: x.get('loss', float('inf')))
                print(f"  Best train loss: {best_train['loss']:.4f} at epoch {best_train['epoch']}")

        # Parse validation results
        valid_results = parse_result_file(os.path.join(save_dir, 'valid_result.txt'))
        if valid_results:
            best_epoch, best_val = find_best_epoch(valid_results, 'recall@5_f')
            print(f"  Validation: {len(valid_results)} epochs evaluated")
            print(f"  Best Recall@5: {best_val:.4f} at epoch {best_epoch}")

            # Print best epoch's full metrics
            best_valid_dict = next((r for r in valid_results if r.get('epoch') == best_epoch), None)
            if best_valid_dict:
                print(f"  Best epoch validation metrics:")
                for k, v in sorted(best_valid_dict.items()):
                    if k != 'epoch':
                        print(f"    {k}: {v:.4f}")

        # Parse test results
        test_results = parse_result_file(os.path.join(save_dir, 'test_result.txt'))
        if test_results:
            # If we have validation, use the best val epoch's test result
            if valid_results and best_epoch:
                best_test = next((r for r in test_results if r.get('epoch') == best_epoch), None)
                if best_test:
                    print(f"  Test metrics (at best val epoch {best_epoch}):")
                    for k, v in sorted(best_test.items()):
                        if k != 'epoch':
                            print(f"    {k}: {v:.4f}")
            else:
                # Otherwise show the last test result
                last_test = test_results[-1]
                print(f"  Test metrics (last epoch {last_test.get('epoch')}):")
                for k, v in sorted(last_test.items()):
                    if k != 'epoch':
                        print(f"    {k}: {v:.4f}")


def compare_variants(base_dir):
    """Create a comparison table across all variants and model types."""
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    # Collect best metrics for each (variant, model_type)
    comparison = []
    for variant in VARIANTS:
        for label, prefix in DIRS:
            save_dir = os.path.join(base_dir, f"{prefix}{variant}")
            if not os.path.exists(save_dir):
                continue

            valid_results = parse_result_file(os.path.join(save_dir, 'valid_result.txt'))
            test_results = parse_result_file(os.path.join(save_dir, 'test_result.txt'))

            if not valid_results:
                continue

            best_epoch, best_r5 = find_best_epoch(valid_results, 'recall@5_f')
            best_valid = next((r for r in valid_results if r.get('epoch') == best_epoch), None)
            best_test = next((r for r in test_results if r.get('epoch') == best_epoch), None) if test_results else None

            if best_valid:
                row = [
                    variant.replace('kuairec_', ''),
                    label,
                    best_epoch,
                    f"{metric_val(best_valid, 'recall@5_f'):.4f}",
                    f"{metric_val(best_valid, 'recall@10_f'):.4f}",
                    f"{metric_val(best_valid, 'recall@20_f'):.4f}",
                    f"{metric_val(best_valid, 'ndcg@10_f'):.4f}",
                    f"{metric_val(best_valid, 'ndcg@20_f'):.4f}",
                    f"{metric_val(best_valid, 'mrr@10_f'):.4f}",
                    f"{metric_val(best_valid, 'ILD@10'):.4f}",
                    f"{metric_val(best_valid, 'CS@10'):.4f}",
                    f"{metric_val(best_valid, 'CC@10'):.4f}",
                ]
                if best_test:
                    row.extend([
                        f"{metric_val(best_test, 'recall@10_f'):.4f}",
                        f"{metric_val(best_test, 'ndcg@10_f'):.4f}",
                        f"{metric_val(best_test, 'ILD@10'):.4f}",
                        f"{metric_val(best_test, 'CS@10'):.4f}",
                        f"{metric_val(best_test, 'CC@10'):.4f}",
                    ])
                else:
                    row.extend(['N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
                comparison.append(row)

    if comparison:
        headers = [
            "Variant", "Model", "Best Ep",
            "V_R@5", "V_R@10", "V_R@20", "V_N@10", "V_N@20", "V_M@10",
            "V_ILD@10", "V_CS@10", "V_CC@10",
            "T_R@10", "T_N@10", "T_ILD@10", "T_CS@10", "T_CC@10"
        ]
        print_table(headers, comparison)
    else:
        print("No validation results found in any directory.")


def main():
    parser = argparse.ArgumentParser(description='Analyze TRIER training results')
    parser.add_argument('--base_dir', default='.', help='Base directory containing save_* folders')
    args = parser.parse_args()

    print(f"Analyzing results in: {os.path.abspath(args.base_dir)}")
    print(f"Variants: {', '.join(VARIANTS)}")
    print(f"Model types: {', '.join(label for label, _ in DIRS)}")

    for variant in VARIANTS:
        analyze_variant(variant, args.base_dir)

    compare_variants(args.base_dir)

    print(f"\n{'='*70}")
    print("Analysis complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
