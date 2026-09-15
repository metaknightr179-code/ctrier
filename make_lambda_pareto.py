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

# Wider figure + generous top margin so legend/caption don't collide
fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.8))
# top=0.70 leaves 30% of figure height for legend + arrow hint + title above the plot
fig.subplots_adjust(left=0.13, right=0.95, top=0.70, bottom=0.18)

# We'll collect x/y limits across all families for tight shared axes
all_xs, all_ys = [], []

for fam in ORDER:
    pts = FAMILIES[fam]
    xs = [p[2] for p in pts]
    ys = [p[1] for p in pts]
    all_xs.extend(xs); all_ys.extend(ys)
    color = PANEL_COLORS[fam]
    hl = PANEL_HIGHLIGHT[fam]

    # thin line connecting all points (lower zorder than points)
    ax.plot(xs, ys, color=color, lw=0.85, alpha=0.60, zorder=3)

    # small filled circles
    ax.scatter(xs, ys, s=14, c=color, zorder=5, edgecolor="white",
               linewidths=0.3, label=fam)

    # arrow on the LAST segment only (direction hint)
    if len(xs) >= 2:
        i = len(xs) - 2
        ax.annotate("", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color=hl, lw=0.8,
                                    mutation_scale=8))

    # Highlight λ=0 start point (open ring) — tiny
    ax.scatter([xs[0]], [ys[0]], s=22, c="none", zorder=6,
               edgecolors=hl, linewidths=0.8)

# ── Axes ────────────────────────────────────────────────────────────────────
ax.set_xlabel("CS@20  (mean adjacent cosine, ↓ better)", labelpad=3)
ax.set_ylabel("NDCG@20  (↑ better)", labelpad=3)

# symlog x-axis: linear for |CS| < 0.01, log beyond.
# This spreads out the dense cluster at CS ∈ [0, 0.02] while still showing
# the λ=0 points at CS ≈ 0.08–0.17 cleanly on the same axis.
import matplotlib.ticker as mticker
ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
ax.set_xlim(-0.002, max(all_xs) * 1.25)

# y-axis: tight
y_lo, y_hi = min(all_ys), max(all_ys)
y_margin = (y_hi - y_lo) * 0.10
ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

# custom x-ticks — show clean labels across the symlog break
ax.set_xticks([0, 0.001, 0.005, 0.01, 0.05, 0.1])
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

# grid — both major + minor so symlog looks intentional
ax.grid(True, linestyle=":", alpha=0.4, zorder=0, which="both")
ax.tick_params(axis="both", which="major", pad=1)
ax.tick_params(axis="x", which="minor", length=1.5)

# spines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ── Legend + direction hint + title — stacked, non-overlapping ──────────
# Plot area uses top=0.70 of figure height. Everything above that is
# figure-level annotation in the empty band [0.70, 1.00].

# 1. Legend: flat row right at top of the empty band
#    We use fig.legend (not ax.legend) so it sits in figure coords cleanly.
leg = fig.legend(labels=ORDER, loc="upper center",
                 bbox_to_anchor=(0.5, 0.975),
                 ncol=4, frameon=False, fontsize=7,
                 handlelength=1.2, handletextpad=0.5, columnspacing=1.5)

# 2. Arrow direction hint — below legend, still well above plot
from matplotlib.patches import FancyArrowPatch
# Use figure fraction coords so it stays aligned regardless of plot layout
arrow = FancyArrowPatch((0.02, 0.90), (0.08, 0.90),
                         transform=fig.transFigure,
                         arrowstyle="-|>", color="gray", lw=0.9,
                         mutation_scale=9, figure=fig)
fig.patches.append(arrow)
fig.text(0.095, 0.895, "λ_c increases  →", fontsize=6, color="gray",
         va="center", style="italic")

# 3. Super title — below the arrow hint, clearly separated
fig.text(0.5, 0.855,
         "Pareto trade-off between repetition (CS@20) and ranking quality (NDCG@20)",
         ha="center", fontsize=7, style="italic")

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
