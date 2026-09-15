"""
Generate bar graphs for the two parameter-sweep tables (tau_s and gamma_o).

Usage: python3 make_param_sweep_bars.py
Outputs:
  figures/tau_s_bars.pdf      -- prospective-intent temperature sweep
  figures/gamma_o_bars.pdf    -- adjacent-order training-weight sweep
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────

# τ_s sweep (prospective-intent temperature) — small-matrix, KuaiRec First-Average
TAU_S = {
    "hparams": ["0.05", "0.1", "0.2", "0.5", "1.0"],
    "HR@5":    [0.1254, 0.1148, 0.1148, 0.1233, 0.1354],
    "HR@10":   [0.1765, 0.1694, 0.1559, 0.1609, 0.1644],
    "HR@20":   [0.2197, 0.2105, 0.1970, 0.1984, 0.2055],
    "NDCG@5":  [0.0646, 0.0768, 0.0710, 0.0736, 0.0683],
    "NDCG@10": [0.0817, 0.0950, 0.0841, 0.0855, 0.0775],
    "NDCG@20": [0.0926, 0.1053, 0.0945, 0.0950, 0.0877],
    "ILD@20":  [1.6310, 1.5946, 1.5781, 1.5799, 1.5609],
    "CC@20":   [0.4018, 0.4370, 0.4006, 0.4101, 0.3940],
    "CS@20":   [0.0020, 0.0029, 0.0068, 0.0152, 0.0308],
    "MaxRun@20": [1.0829, 1.0843, 1.1836, 1.4096, 1.8306],
}

# γ_o sweep (adjacent-order training weight) — small-matrix, PACER KuaiRec First-Average
GAMMA_O = {
    "hparams": ["0", "0.001", "0.005", "0.01", "0.05", "0.1"],
    "HR@5":    [0.1212, 0.1141, 0.1191, 0.1205, 0.1247, 0.1106],
    "HR@10":   [0.1722, 0.1609, 0.1694, 0.1736, 0.1701, 0.1595],
    "HR@20":   [0.2105, 0.1935, 0.2225, 0.2190, 0.2261, 0.2105],
    "NDCG@5":  [0.0792, 0.0849, 0.0723, 0.0983, 0.0801, 0.0547],
    "NDCG@10": [0.0963, 0.1000, 0.0888, 0.1156, 0.0948, 0.0708],
    "NDCG@20": [0.1058, 0.1083, 0.1022, 0.1271, 0.1088, 0.0836],
    "ILD@20":  [1.5933, 1.6174, 1.6063, 1.5881, 1.6019, 1.5984],
    "CC@20":   [0.4354, 0.4222, 0.4392, 0.4277, 0.4344, 0.3829],
    "CS@20":   [0.1080, 0.1335, 0.1218, 0.1675, 0.1081, 0.1122],
    "MaxRun@20": [2.6577, 2.7945, 2.9752, 4.2261, 2.6832, 2.6421],
}

# ── Plot helpers ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

ACC_METRICS = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
DIV_METRICS = ["ILD@20", "CC@20"]
REP_METRICS = ["CS@20", "MaxRun@20"]   # lower is better
COLORS_ACC = ["#3b82f6", "#2563eb", "#1d4ed8", "#f59e0b", "#d97706", "#b45309"]
COLORS_DIV = ["#10b981", "#059669"]
COLORS_REP = ["#ef4444", "#dc2626"]

def sweep_bar(sweep, title, xlabel, out_path):
    hparams = sweep["hparams"]
    x = np.arange(len(hparams))
    w = 0.28        # bar width
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4))
    fig.suptitle(title, y=1.02)

    # Accuracy block (top panel gets more visual emphasis)
    ax = axes[0]
    for i, m in enumerate(ACC_METRICS):
        ax.bar(x + i*w - 2.5*w, sweep[m], width=w, label=m,
               color=COLORS_ACC[i], alpha=0.85)
    ax.set_xticks(x + 0.5*w)
    ax.set_xticklabels(hparams, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title("Accuracy")
    ax.legend(ncol=2, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # Diversity block
    ax = axes[1]
    for i, m in enumerate(DIV_METRICS):
        ax.bar(x + i*w - 0.5*w, sweep[m], width=w, label=m,
               color=COLORS_DIV[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(hparams, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title("Diversity")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # Repetition block — lower is better; mark the ideal direction on axis label
    ax = axes[2]
    for i, m in enumerate(REP_METRICS):
        ax.bar(x + i*w - 0.5*w, sweep[m], width=w, label=m,
               color=COLORS_REP[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(hparams, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title("Repetition (lower is better)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
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
