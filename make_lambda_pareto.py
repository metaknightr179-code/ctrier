"""
Pareto curves: NDCG@20 (↑ accuracy) vs CS@20 (↓ repetition)
from the λ_c inference-time penalty sweep.

Layout: 1 row × 4 panels — TRIER, TRIER-C, TRIER-L, PACER-Full.
Each panel shows 6 points (λ_c = 0, 0.001, 0.005, 0.01, 0.05, 0.1)
connected by an arrowed line from λ_c=0 → λ_c=0.1 (the direction of
increasing penalty strength). The arrow makes it visually obvious that
increasing λ_c moves the trade-off toward lower CS (better diversity)
with some loss in NDCG.

Panel layout identical to make_param_sweep_bars.py — serif fonts, 7 pt,
tight spacing, legends above panels (but here we label each point with
its λ_c value, so no legend needed; just panel titles).

Usage: python3 make_lambda_pareto.py
Output: figures/lambda_c_pareto.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ── Data — λ_c sweep (NDCG@20, CS@20) per family ────────────────────────────
# Each tuple: (lambda_c_label, NDCG@20, CS@20)
# Sorted by λ_c ascending so the arrow flows naturally from λ=0 → λ=0.1

FAMILIES = {
    "TRIER": [          # no-type, γ=0 base (no L_order training)
        ("0",     0.0972, 0.0975),
        ("0.001", 0.0970, 0.0187),
        ("0.005", 0.0966, 0.0017),
        ("0.01",  0.0966, 0.0001),
        ("0.05",  0.0966, 0.0000),
        ("0.1",   0.0966, 0.0000),
    ],
    "TRIER-C": [        # content, γ=0
        ("0",     0.1058, 0.1080),
        ("0.001", 0.1057, 0.0196),
        ("0.005", 0.1053, 0.0084),
        ("0.01",  0.1053, 0.0029),
        ("0.05",  0.1053, 0.0000),
        ("0.1",   0.1053, 0.0000),
    ],
    "TRIER-L": [        # no-type, γ=0.01 (L_order trained, frustrated regime at λ_c=0)
        ("0",     0.0670, 0.0787),
        ("0.001", 0.0676, 0.0093),
        ("0.005", 0.0681, 0.0031),
        ("0.01",  0.0681, 0.0002),
        ("0.05",  0.0681, 0.0000),
        ("0.1",   0.0681, 0.0000),
    ],
    "PACER-Full": [     # content + γ=0.01 (L_order trained, frustrated regime at λ_c=0)
        ("0",     0.1271, 0.1675),
        ("0.001", 0.1272, 0.0607),
        ("0.005", 0.1265, 0.0122),
        ("0.01",  0.1262, 0.0008),
        ("0.05",  0.1261, 0.0000),
        ("0.1",   0.1261, 0.0000),
    ],
}

# ── Panel colors — TRIER (no L_order) vs TRIER-L/PACER-Full (L_order trained)
# Green = base γ=0 families (TRIER, TRIER-C): no frustrated regime
# Red   = L_order-trained families (TRIER-L, PACER-Full): frustrated at λ_c=0

PANEL_COLORS = {
    "TRIER":      "#2563eb",   # blue
    "TRIER-C":    "#059669",   # green-emerald
    "TRIER-L":    "#dc2626",   # red (frustrated at λ_c=0)
    "PACER-Full": "#9333ea",   # purple (frustrated at λ_c=0)
}

PANEL_HIGHLIGHT = {
    "TRIER":      "#1e40af",
    "TRIER-C":    "#047857",
    "TRIER-L":    "#b91c1c",
    "PACER-Full": "#7c3aed",
}

# ── Plot config (match make_param_sweep_bars.py) ──────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size":   7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "figure.dpi": 200,
    "savefig.dpi": 400,
})

ORDER = ["TRIER", "TRIER-C", "TRIER-L", "PACER-Full"]

fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.6),
                         sharey=False, sharex=False)
fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.18,
                    wspace=0.38, hspace=0.0)

for ax_idx, fam in enumerate(ORDER):
    ax = axes[ax_idx]
    pts = FAMILIES[fam]
    xs = [p[2] for p in pts]   # CS@20 (→)
    ys = [p[1] for p in pts]   # NDCG@20 (↑)
    labels = [p[0] for p in pts]
    color = PANEL_COLORS[fam]
    hl = PANEL_HIGHLIGHT[fam]

    # Draw connected segments with arrow
    # Plot points (filled circles)
    ax.scatter(xs, ys, s=28, c=color, zorder=5, edgecolor="white",
               linewidths=0.5)

    # Draw line with arrows between consecutive points
    for i in range(len(xs) - 1):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        # Draw arrow only on the LAST segment of each family (λ_c=0.05 → 0.1)
        # to keep it clean
        if i == len(xs) - 2:
            ax.annotate("", xy=(xs[i+1], ys[i+1]),
                        xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=hl,
                                        lw=0.9,
                                        mutation_scale=10))
        # thin line connecting all points
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=color, lw=0.7, alpha=0.6, zorder=3)

    # Highlight the FIRST point (λ_c=0 — the "frustrated" start)
    ax.scatter([xs[0]], [ys[0]], s=42, c="none", zorder=6,
               edgecolors=hl, linewidths=1.0)
    ax.annotate("λ=0", xy=(xs[0], ys[0]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=5, color=hl, fontweight="bold")

    # Highlight the LAST point (λ_c=0.1 — the converged end)
    ax.scatter([xs[-1]], [ys[-1]], s=42, c="none", zorder=6,
               edgecolors=hl, linewidths=1.0)
    ax.annotate("λ=0.1", xy=(xs[-1], ys[-1]),
                xytext=(-4, -10), textcoords="offset points",
                fontsize=5, color=hl, fontweight="bold",
                ha="right")

    # Axes
    ax.set_title(fam, fontweight="bold", pad=3)
    ax.set_xlabel("CS@20 (↓ better)", labelpad=2)
    if ax_idx == 0:
        ax.set_ylabel("NDCG@20 (↑ better)", labelpad=2)

    # x-axis: log scale? No — values are 0 to 0.2. But CS=0.0000 values
    # will sit at x=0 which is hard to distinguish from x=0.0001.
    # Use a small xlim to show the dense cluster clearly + show the λ=0 point.
    # Actually let's just set tight limits
    ax.set_xlim(-0.005, max(xs) * 1.15 if max(xs) > 0 else 0.01)

    # y-axis: tight
    y_margin = (max(ys) - min(ys)) * 0.08 if max(ys) > min(ys) else 0.002
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

    # grid
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.tick_params(axis="both", which="major", pad=1)

    # spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# Super title row: brief note
fig.text(0.5, 0.91,
         "Pareto frontier: λ_c (inference-time penalty) trades off repetition (CS@20) for accuracy (NDCG@20)",
         ha="center", va="center", fontsize=6.5, style="italic")

# Arrow legend: arrow direction = increasing λ_c
# Use a FancyArrowPatch positioned in figure coordinates
from matplotlib.patches import FancyArrowPatch
arrow = FancyArrowPatch((0.02, 0.86), (0.07, 0.86),
                         transform=fig.transFigure,
                         arrowstyle="-|>", color="gray", lw=1.0,
                         mutation_scale=10, figure=fig)
fig.patches.append(arrow)
fig.text(0.085, 0.855, "λ_c increases →", fontsize=5.5, color="gray",
         va="center", style="italic")

out_path = os.path.join(OUT, "lambda_c_pareto.pdf")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path}")

# Also print a quick numeric summary
print("\nPareto summary (NDCG drop per decade of CS):")
print("-" * 70)
print(f"{'Family':<14} {'NDCG drop at CS=0 (λ=0→0.01)':>30} {'CS drop ratio (λ=0 / λ=0.01)':>25}")
print("-" * 70)
for fam in ORDER:
    p000 = dict([(p[0], p) for p in FAMILIES[fam]])["0"]
    p001 = dict([(p[0], p) for p in FAMILIES[fam]])["0.01"]
    ndcg_drop = p000[1] - p001[1]
    cs_drop   = p000[2] / max(p001[2], 1e-10)
    print(f"{fam:<14} {ndcg_drop:>0.4f}                         {cs_drop:>8.0f}×")
