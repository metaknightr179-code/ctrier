"""
Generate bar graphs for the two parameter-sweep tables.

Layout per sweep: 1 row × 5 columns
  Panel 1: HR@5/10/20      (accuracy, ↑)
  Panel 2: NDCG@5/10/20     (ranking accuracy, ↑)
  Panel 3: ILD@20 + CC@20   (set-level diversity, ↑)
  Panel 4: CS@20            (adjacent similarity, ↓)
  Panel 5: MaxRun@20        (contiguous runs, ↓)

Legends live OUTSIDE each panel (above, center), tiny font (6 pt), no border.
All subplots share the same x-axis (hyperparameter values).

Usage: python3 make_param_sweep_bars.py
Outputs:
  figures/tau_s_bars.pdf    (τ_s prospective-intent temperature)
  figures/gamma_o_bars.pdf  (γ_o training weight, PACER)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────

TAU_S = {
    "hparams": ["0.05", "0.1", "0.2", "0.5", "1.0"],
    "HR@5":       [0.1254, 0.1148, 0.1148, 0.1233, 0.1354],
    "HR@10":      [0.1765, 0.1694, 0.1559, 0.1609, 0.1644],
    "HR@20":      [0.2197, 0.2105, 0.1970, 0.1984, 0.2055],
    "NDCG@5":     [0.0646, 0.0768, 0.0710, 0.0736, 0.0683],
    "NDCG@10":    [0.0817, 0.0950, 0.0841, 0.0855, 0.0775],
    "NDCG@20":    [0.0926, 0.1053, 0.0945, 0.0950, 0.0877],
    "ILD@20":     [1.6310, 1.5946, 1.5781, 1.5799, 1.5609],
    "CC@20":      [0.4018, 0.4370, 0.4006, 0.4101, 0.3940],
    "CS@20":      [0.0020, 0.0029, 0.0068, 0.0152, 0.0308],
    "MaxRun@20":  [1.0829, 1.0843, 1.1836, 1.4096, 1.8306],
}

GAMMA_O = {
    "hparams": ["0", "0.001", "0.005", "0.01", "0.05", "0.1"],
    "HR@5":       [0.1212, 0.1141, 0.1191, 0.1205, 0.1247, 0.1106],
    "HR@10":      [0.1722, 0.1609, 0.1694, 0.1736, 0.1701, 0.1595],
    "HR@20":      [0.2105, 0.1935, 0.2225, 0.2190, 0.2261, 0.2105],
    "NDCG@5":     [0.0792, 0.0849, 0.0723, 0.0983, 0.0801, 0.0547],
    "NDCG@10":    [0.0963, 0.1000, 0.0888, 0.1156, 0.0948, 0.0708],
    "NDCG@20":    [0.1058, 0.1083, 0.1022, 0.1271, 0.1088, 0.0836],
    "ILD@20":     [1.5933, 1.6174, 1.6063, 1.5881, 1.6019, 1.5984],
    "CC@20":      [0.4354, 0.4222, 0.4392, 0.4277, 0.4344, 0.3829],
    "CS@20":      [0.1080, 0.1335, 0.1218, 0.1675, 0.1081, 0.1122],
    "MaxRun@20":  [2.6577, 2.7945, 2.9752, 4.2261, 2.6832, 2.6421],
}

# ── Plot config ─────────────────────────────────────────────────────────────

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

HR_METS     = ["HR@5",   "HR@10",   "HR@20"]
NDCG_METS   = ["NDCG@5", "NDCG@10", "NDCG@20"]
DIV_METS    = ["ILD@20", "CC@20"]
REP_CS      = ["CS@20"]
REP_MAXRUN  = ["MaxRun@20"]

COLORS_HR   = ["#3b82f6", "#2563eb", "#1d4ed8"]       # blues → darker at larger k
COLORS_NDCG = ["#f97316", "#ea580c", "#c2410c"]       # oranges
COLORS_DIV  = ["#10b981", "#059669"]                   # greens
COLORS_CS   = ["#ef4444"]                              # single red
COLORS_MR   = ["#8b5cf6"]                              # single purple

BAR_W = 0.22   # width per bar within a hparam group

def sweep_bar(sweep, title, xlabel, out_path):
    hparams = sweep["hparams"]
    N = len(hparams)
    x = np.arange(N)

    fig, axes = plt.subplots(1, 5, figsize=(8.5, 2.2), constrained_layout=True)
    fig.suptitle(title, fontsize=8, y=1.06)

    def plot_group(ax, mets, colors, panel_title, ylabel_tip=None):
        # Center bars around x ticks
        m = len(mets)
        widths = np.arange(m) - (m - 1) / 2.0   # e.g. [-1, 0, 1] for m=3
        bars = []
        labels_out = []
        for i, (met, c) in enumerate(zip(mets, colors)):
            b = ax.bar(x + widths[i] * BAR_W, sweep[met], width=BAR_W,
                       color=c, alpha=0.85, label=met)
            bars.append(b)
            labels_out.append(met)
        ax.set_xticks(x)
        ax.set_xticklabels(hparams, fontsize=6, rotation=0)
        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_title(panel_title, fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        # Thin frame
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        # Legend OUTSIDE panel: above, centered, tiny font, no frame
        ax.legend(bars, labels_out,
                  loc="upper center", bbox_to_anchor=(0.5, 1.32),
                  ncol=min(m, 4), frameon=False, handlelength=1.2,
                  columnspacing=0.6, handletextpad=0.3,
                  borderaxespad=0.0)
        if ylabel_tip:
            ax.set_ylabel(ylabel_tip, fontsize=6, color="#555555")

    plot_group(axes[0], HR_METS,     COLORS_HR,   "Recall (HR↑)")
    plot_group(axes[1], NDCG_METS,   COLORS_NDCG, "NDCG (↑)")
    plot_group(axes[2], DIV_METS,    COLORS_DIV,  "Set diversity (↑)")
    plot_group(axes[3], REP_CS,      COLORS_CS,   "Adjacent sim (CS↓)", "lower is better")
    plot_group(axes[4], REP_MAXRUN,  COLORS_MR,   "MaxRun (↓)",         "lower is better")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")

# ── Generate ────────────────────────────────────────────────────────────────
sweep_bar(TAU_S,
          r"Sensitivity to prospective-intent temperature $\tau_s$",
          r"$\tau_s$",
          os.path.join(OUT, "tau_s_bars.pdf"))

sweep_bar(GAMMA_O,
          r"Sensitivity to adjacent-order training weight $\gamma_o$ (PACER)",
          r"$\gamma_o$",
          os.path.join(OUT, "gamma_o_bars.pdf"))

print("Done.")
