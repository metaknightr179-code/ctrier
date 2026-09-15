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
import matplotlib.ticker as mticker
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

# TIGHT linear x-axis on the repair region. The λ_c=0 outlier points sit at
# CS ∈ [0.08, 0.17] (far right) and are NOT drawn on the main plot — instead
# we show them as dashed lead-in arrows so the *repair dynamics* (the actual
# story) are not compressed into invisibility.
X_MAX = 0.025   # covers all λ≥0.001 points with room to breathe
ax.set_xlim(0, X_MAX)

# y-axis: tight on the NDCG range of the plotted points
# Exclude the extreme TRIER-L λ=0 NDCG=0.0670 outlier when computing range
# (we don't plot it on-axis anyway)
plotted_ys = []
for fam in ORDER:
    pts = FAMILIES[fam]
    plotted_ys.extend([p[1] for p in pts if p[2] <= X_MAX])
y_lo, y_hi = min(plotted_ys), max(plotted_ys)
y_span = y_hi - y_lo
y_margin = y_span * 0.15
ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

# x-ticks — clean, readable
ax.set_xticks([0, 0.005, 0.01, 0.015, 0.02])
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

# grid
ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
ax.tick_params(axis="both", which="major", pad=1)

# spines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ── λ=0 LEAD-IN ARROWS ───────────────────────────────────────────────────
# For each family, draw a thin dashed arrow that starts at the right edge of
# the plot (X_MAX, that family's last-plotted-point NDCG) and goes to the
# actual λ=0 location (CS=x0, NDCG=y0). Arrow label says "λ=0".
# This communicates that the pre-penalty state is further right than shown.
from matplotlib.patches import FancyArrowPatch as FAP
for fam in ORDER:
    pts = FAMILIES[fam]
    x0, y0 = pts[0][2], pts[0][1]          # λ_c=0  (off-axis, far right)
    # find the rightmost point we *do* plot (largest CS ≤ X_MAX)
    plotted = [(p[2], p[1]) for p in pts if p[2] <= X_MAX]
    if not plotted:
        continue
    x_plot, y_plot = plotted[-1]            # e.g. (0.02, NDCG at λ=0.1)
    color = PANEL_HIGHLIGHT[fam]

    # Dashed connector from (X_MAX, y_plot) → (x0, y0), clipped by xlim
    # We'll manually clip it by only drawing up to X_MAX
    ax.annotate("",
                xy=(x0, y0),                # λ=0 destination (off-axis right)
                xytext=(X_MAX, y_plot),       # start at right edge of plot
                xycoords="data",
                arrowprops=dict(arrowstyle="-",
                                color=color,
                                lw=0.8,
                                linestyle="--",
                                alpha=0.7))
    # tiny text at (X_MAX + 0.0005, y_plot) saying "λ=0→"
    ax.text(X_MAX + 0.0002, y_plot, "λ=0→", fontsize=5,
            color=color, va="center", style="italic")

# ── Legend + direction hint + title — stacked, non-overlapping ──────────

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
