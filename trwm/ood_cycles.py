"""Out-of-distribution recursion-depth sweep (inference-only).

Each trained recursive model is evaluated at recursion depths other than the one it
was trained at (SimpleTRM/SRM x {1h1l, 4h3l}); x is the total inner iteration count
H*L. The in-distribution (trained) depth is marked with a ringed point, so departure
from it reads as OOD generalisation over compute.

Encoding matches simple_v_srm_depth.py: colour = trained depth (1h1l dark, 4h3l light),
line style = architecture (SimpleTRM solid, SRM dashed). Reward is the mean DMC episode
return over the ~21 eval tasks (each ~0-1000), averaged over 3 seeds; band = sd over
seeds.

Data: analysis/results/ood_cycles.csv in the newt repo (produced by
analysis/sweep_ood_cycles.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fanda.palettes import blue_rocket
from fanda.visualizations import annotate_axis, decorate_axis, subplots
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

FIG_WIDTH_IN = 3.073   # axes box ~2.161x1.662in (single panel), matching srm_cycles.py
FIG_HEIGHT_IN = 2.348
FONT_PT = 11

CSV = Path("/home/holmes/projects/thesis/vla/newt/analysis/results/ood_cycles.csv")

blues = blue_rocket(3)
BASE_COLOR = {1: blues[0], 4: blues[2]}          # keyed by base_H (1h1l dark, 4h3l light)
ARCH_DASH = {"simple": "", "srm": (4, 1.5)}
ARCH_NAME = {"simple": "SimpleTRM", "srm": "SRM"}
MODEL_ORDER = ["SimpleTRM 1h1l", "SimpleTRM 4h3l", "SRM 1h1l", "SRM 4h3l"]

df = pd.read_csv(CSV)

# per (model, seed, depth): mean reward over tasks; then mean/sd over seeds.
per_seed = (df.groupby(["model", "arch", "base_H", "eval_H", "eval_L", "HL", "is_base",
                        "seed"])["episode_reward"].mean().reset_index())
agg = (per_seed.groupby(["model", "arch", "base_H", "eval_H", "eval_L", "HL", "is_base"])
       ["episode_reward"].agg(["mean", "std"]).reset_index())

fanda = subplots(1, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
ax = fanda.ax

for model in MODEL_ORDER:
    sub = agg[agg["model"] == model].sort_values("HL")
    if sub.empty:
        continue
    arch = sub["arch"].iloc[0]
    color = BASE_COLOR[int(sub["base_H"].iloc[0])]
    dash = ARCH_DASH[arch]
    x, y = sub["HL"].to_numpy(), sub["mean"].to_numpy()
    e = np.nan_to_num(sub["std"].to_numpy())   # single-seed cells have NaN std
    ls = (0, dash) if dash else "-"
    ax.plot(x, y, ls=ls, color=color, lw=1.6, marker="o", ms=3, zorder=3)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, lw=0)
    # ring the in-distribution (trained) depth
    base = sub[sub["is_base"] == 1]
    if not base.empty:
        ax.plot(base["HL"], base["mean"], marker="o", ms=8, mfc="none",
                mec=color, mew=1.6, ls="none", zorder=4)

annotate_axis(fanda, xlabel="Recursion depth (H·L inner cycles)",
              ylabel="Episode Reward", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
ax.set_xticks(sorted(agg["HL"].unique()))

plt.tight_layout()

handles = [Line2D([], [], color=BASE_COLOR[b], lw=1.6, label=f"{b}h.. base")
           for b in (1, 4)]
handles = [
    Line2D([], [], color=blues[0], lw=1.6, label="1h1l"),
    Line2D([], [], color=blues[2], lw=1.6, label="4h3l"),
    Line2D([], [], color="#444444", dashes=ARCH_DASH["simple"] or (None, None),
           lw=1.6, label="SimpleTRM"),
    Line2D([], [], color="#444444", dashes=ARCH_DASH["srm"], lw=1.6, label="SRM"),
    Line2D([], [], color="#444444", marker="o", ms=7, mfc="none", mew=1.4,
           ls="none", label="in-dist."),
]
fanda.fig.legend(handles, [h.get_label() for h in handles],
                 loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                 fontsize=FONT_PT - 2, fancybox=True, handlelength=1.5,
                 columnspacing=1.0, handletextpad=0.5)

save_fig(fanda, name="images/ood_cycles", format="svg")
close_fig(fanda)
