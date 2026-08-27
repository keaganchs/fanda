"""SimpleTRM skip-connection ablation on PushT -- 3-seed version.

Same 4-panel layout as abl_skip_pusht.py, but bands are +/-1 SD over the 3
training seeds (the reference's seed-pooled uncertainty), not over eval envs /
diagnostic batches. Seed 0 reuses the single-seed runs; seeds 1-2 are the
skab_<tag>_s<seed> checkpoints.

Data (newt analysis/results/, from diag_skip_ablation_3seed.py):
  skip_ablation_reward_3seed.csv, skip_ablation_diag_3seed.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fanda.visualizations import annotate_axis, decorate_axis, subplots
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

FIG_WIDTH_IN, FIG_HEIGHT_IN, FONT_PT = 5.874, 4.990, 11
H, L = 4, 3
RESULTS = Path("/home/holmes/projects/thesis/vla/newt/analysis/results")

SOLID, DASH = "-", (0, (4, 2))
STYLE = {
    "none":       ("No skip",              "#333333", SOLID),
    "additive":   ("Additive",             "#D55E00", SOLID),
    "zero_scale": ("Additive (zero-init)", "#D55E00", DASH),
    "mlp":        ("MLP",                  "#0072B2", SOLID),
    "zero_mlp":   ("MLP (zero-init)",      "#0072B2", DASH),
    "mgr":        ("MGR",                  "#009E73", SOLID),
}
ORDER = ["none", "additive", "zero_scale", "mlp", "zero_mlp", "mgr"]

rew = pd.read_csv(RESULTS / "skip_ablation_reward_3seed.csv")
diag = pd.read_csv(RESULTS / "skip_ablation_diag_3seed.csv")


def seed_band(df, group_cols, val="value"):
    """mean +/- SD over seeds, after collapsing within-seed replicates."""
    per_seed = df.groupby(group_cols + ["seed"])[val].mean().reset_index()
    return (per_seed.groupby(group_cols)[val]
            .agg(mean="mean", sd="std").reset_index())


fanda = subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))


def draw(ax, skip, x, y, sd=None, ms=3, clip=False):
    _, c, ls = STYLE[skip]
    ax.plot(x, y, color=c, ls=ls, lw=1.4, marker="o", ms=ms, zorder=3)
    if sd is not None:
        lo = np.clip(y - sd, 1e-12, None) if clip else y - sd
        ax.fill_between(x, lo, y + sd, color=c, alpha=0.12, lw=0)


# --- panel 1: reward vs epoch (band = SD over seeds) ---
fanda.select(0)
ax = fanda.ax
rc = seed_band(rew, ["skip", "epoch"], val="episode_reward")
rc["mean_k"], rc["sd_k"] = rc["mean"] / 1e3, rc["sd"].fillna(0) / 1e3
for skip in ORDER:
    s = rc[rc["skip"] == skip].sort_values("epoch")
    if not s.empty:
        draw(ax, skip, s["epoch"].to_numpy(), s["mean_k"].to_numpy(), s["sd_k"].to_numpy())
annotate_axis(fanda, xlabel="Training epoch", ylabel="Episode Reward ($\\times 10^3$)",
              title="Episode Reward", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

# --- panel 2: gradient norm z_0..z_{L-1}, y (band = SD over seeds) ---
fanda.select(1)
ax = fanda.ax
gcols = [f"z_{i}" for i in range(L)] + ["y"]
xg = np.arange(len(gcols))
g = seed_band(diag[diag["kind"] == "gradnorm"], ["skip", "idx"]).set_index(["skip", "idx"])
for skip in ORDER:
    if (skip, "y") not in g.index:
        continue
    m = np.array([g.loc[(skip, c), "mean"] for c in gcols])
    sd = np.array([np.nan_to_num(g.loc[(skip, c), "sd"]) for c in gcols])
    draw(ax, skip, xg, m, sd, clip=True)
ax.set_yscale("log")
ax.set_xticks(xg)
ax.set_xticklabels([r"$z_0$", r"$z_1$", r"$z_2$", r"$y$"])
annotate_axis(fanda, xlabel="Recursion cycle", ylabel="Gradient norm",
              title="Gradient Norm: z & y carries", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

# --- panel 3: advantage margin per cycle (mean over seeds) ---
fanda.select(2)
ax = fanda.ax
mm = diag[diag["kind"] == "margin"].copy()
mm["idx"] = mm["idx"].astype(int)
mb = seed_band(mm, ["skip", "idx"])
for skip in ORDER:
    s = mb[mb["skip"] == skip].sort_values("idx")
    if not s.empty:
        _, c, ls = STYLE[skip]
        ax.plot(s["idx"].to_numpy(), s["mean"].to_numpy(), color=c, ls=ls,
                lw=1.4, marker="o", ms=2.5, zorder=3)
ax.axhline(0, color="#999999", lw=0.8, ls=":", zorder=1)
ax.set_yscale("symlog", linthresh=1.0)
annotate_axis(fanda, xlabel="Recursion cycle ($h\\cdot L + l$)",
              ylabel="Advantage margin $A_i$",
              title="Advantage Margin per Cycle", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

# --- panel 4: high-carry update per outer cycle (band = SD over seeds) ---
fanda.select(3)
ax = fanda.ax
dd = diag[diag["kind"] == "ydelta"].copy()
dd["idx"] = dd["idx"].astype(int)
db = seed_band(dd, ["skip", "idx"])
for skip in ORDER:
    s = db[db["skip"] == skip].sort_values("idx")
    if s.empty:
        continue
    draw(ax, skip, s["idx"].to_numpy(), s["mean"].to_numpy(),
         np.nan_to_num(s["sd"].to_numpy()), clip=True)
ax.set_yscale("log")
ax.set_xticks(np.arange(H))
annotate_axis(fanda, xlabel="Outer cycle $h$", ylabel=r"$\|y_i - y_{i-1}\|$",
              title="High-carry Update per Cycle", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

plt.tight_layout()
handles = [Line2D([], [], color=STYLE[s][1], ls=STYLE[s][2], lw=1.8, label=STYLE[s][0])
           for s in ORDER]
fanda.fig.legend(handles, [STYLE[s][0] for s in ORDER],
                 loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                 fontsize=FONT_PT - 1, fancybox=True, handlelength=1.8,
                 columnspacing=1.1, handletextpad=0.5)
save_fig(fanda, name="images/abl_skip_pusht_3seed", format="svg")
close_fig(fanda)
