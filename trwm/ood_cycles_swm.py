"""OOD recursion-depth sweep on stable-worldmodel PushT (inference-only).

Simplified SimpleTRM / SRM dynamics heads inside the stable-worldmodel TD-MPC2,
trained offline on the pusht_srm dataset at 4h3l (or with uniformly sampled
cycles H~U[1,4], L~U[1,6]) and evaluated with CEM planning at recursion depths
they were not trained at. 2x2 panels: architecture x skip type. Colour encodes
the training regime (fixed 4h3l dark, uniform light); the ringed point marks
the in-distribution depth of the fixed models (uniform models have no single
trained depth, so no ring).

Reward is the PushT episode return (sum of -distance per step, /10^3), mean
over 2 episode batches; band = sd over the 10 eval envs ("seed" column).

Data: analysis/results/ood_cycles_swm.csv in the newt repo (produced by
stable-worldmodel/scripts/plan/sweep_ood_recursion.py).
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

FIG_WIDTH_IN = 5.6
FIG_HEIGHT_IN = 4.6
FONT_PT = 11

CSV = Path("/home/holmes/projects/thesis/vla/newt/analysis/results/ood_cycles_swm.csv")

blues = blue_rocket(3)
REGIME_COLOR = {"fix": blues[0], "uni": blues[2]}
REGIME_NAME = {"fix": "Trained 4h3l", "uni": "Uniform cycles"}

# panel grid: rows = skip, cols = arch
ARCH_ORDER = ["SimpleTRM", "SRM"]
SKIP_ORDER = ["none", "zmlp"]
SKIP_NAME = {"none": "no skip", "zmlp": "zero-init MLP skip"}

df = pd.read_csv(CSV)
parts = df["model"].str.split(" ", expand=True)
df["arch_name"], df["skip"], df["regime"] = parts[0], parts[1], parts[2]
df["reward_k"] = df["episode_reward"] / 1e3

# per (model, depth, seed): mean over episode batches; then mean/sd over seeds
per_seed = (df.groupby(["model", "arch_name", "skip", "regime", "eval_H",
                        "eval_L", "HL", "is_base", "seed"])["reward_k"]
            .mean().reset_index())
agg = (per_seed.groupby(["model", "arch_name", "skip", "regime", "eval_H",
                         "eval_L", "HL", "is_base"])["reward_k"]
       .agg(["mean", "std"]).reset_index())

fanda = subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                 sharex=True, sharey=True)
for i, (skip, arch) in enumerate(
        [(s, a) for s in SKIP_ORDER for a in ARCH_ORDER]):
    fanda.select(i)
    ax = fanda.ax
    for regime in ["fix", "uni"]:
        sub = (agg[(agg["arch_name"] == arch) & (agg["skip"] == skip)
                   & (agg["regime"] == regime)].sort_values("HL"))
        if sub.empty:
            continue
        color = REGIME_COLOR[regime]
        x, y = sub["HL"].to_numpy(), sub["mean"].to_numpy()
        e = np.nan_to_num(sub["std"].to_numpy())
        ax.plot(x, y, "-", color=color, lw=1.6, marker="o", ms=3, zorder=3)
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, lw=0)
        if regime == "fix":
            base = sub[sub["is_base"] == 1]
            ax.plot(base["HL"], base["mean"], marker="o", ms=8, mfc="none",
                    mec=color, mew=1.6, ls="none", zorder=4)
    annotate_axis(
        fanda,
        xlabel="Recursion depth (H·L inner cycles)" if i >= 2 else "",
        ylabel="Episode Reward ($\\times 10^3$)" if i % 2 == 0 else "",
        title=f"{arch}: {SKIP_NAME[skip]}", labelsize=FONT_PT,
    )
    decorate_axis(fanda, ticklabelsize=FONT_PT,
                  spines=["top", "right", "bottom", "left"])

fanda.axs.flat[0].set_xticks(sorted(agg["HL"].unique()))

plt.tight_layout()

handles = [
    Line2D([], [], color=REGIME_COLOR["fix"], lw=1.6, marker="o", ms=3,
           label=REGIME_NAME["fix"]),
    Line2D([], [], color=REGIME_COLOR["uni"], lw=1.6, marker="o", ms=3,
           label=REGIME_NAME["uni"]),
    Line2D([], [], color="#444444", marker="o", ms=7, mfc="none", mew=1.4,
           ls="none", label="in-dist."),
]
fanda.fig.legend(handles, [h.get_label() for h in handles],
                 loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                 fontsize=FONT_PT - 1, fancybox=True, handlelength=1.5,
                 columnspacing=1.0, handletextpad=0.5)

save_fig(fanda, name="images/ood_cycles_swm", format="svg")
close_fig(fanda)
