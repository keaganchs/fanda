"""OOD recursion-depth extrapolation with robustness methods (stable-worldmodel PushT).

Companion to ood_cycles_swm.py. Each recursive dynamics head ({SimpleTRM, SRM},
skip=none) is trained with uniform depth sampling; on top of that we add one of
the OOD-cycle robustness methods and evaluate CEM-planning episode reward at
recursion depths H.L not seen at (the nominal) training depth 4h3l:

  * uni       -- uniform sampling only (baseline)
  * ds        -- + deep supervision (per-cycle consistency)
  * spec      -- + spectral norm (Lipschitz / non-expansive core)
  * specguid  -- + spectral norm + GRAM-style stochastic guidance
  * film      -- + FiLM conditioning on cycle index and total depth
  * all       -- all of the above

Two columns (architecture). Top row: reward vs recursion depth; band = standard
error of the mean over the 10 eval envs (NOT the raw sd -- start-state variance
dominates and swamps the method effect, so the honest comparison is paired).
Bottom row: paired improvement over the uniform baseline, matched on start state
(depth x env x batch); bars are the paired mean delta, whiskers its standard
error. This is where the method effect is actually resolved.

Data: analysis/results/ood_cycles_methods.csv in the newt repo (produced by
stable-worldmodel/scripts/plan/sweep_ood_methods.py).
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

FIG_WIDTH_IN = 6.6
FIG_HEIGHT_IN = 5.2
FONT_PT = 11

CSV = Path("/home/holmes/projects/thesis/vla/newt/analysis/results/ood_cycles_methods.csv")
BASE_HL = 12  # nominal training depth (4h x 3l)

ARCH_ORDER = ["SimpleTRM", "SRM"]
METHOD_ORDER = ["uni", "ds", "spec", "specguid", "film", "all"]
METHOD_NAME = {
    "uni": "Uniform (baseline)", "ds": "+ Deep supervision",
    "spec": "+ Spectral norm", "specguid": "+ Spectral + guidance",
    "film": "+ FiLM cycles", "all": "+ All",
}
# Okabe-Ito (colorblind-safe); baseline grey + dashed.
METHOD_COLOR = {
    "uni": "#666666", "ds": "#0072B2", "spec": "#E69F00",
    "specguid": "#D55E00", "film": "#009E73", "all": "#CC79A7",
}
METHOD_STYLE = {"uni": (0, (4, 2))}

df = pd.read_csv(CSV)
parts = df["model"].str.split(" ", expand=True)
df["arch_name"], df["method"] = parts[0], parts[1]
df["r"] = df["episode_reward"] / 1e3

# --- top row data: per-depth mean +/- se over eval envs ---
per_seed = (df.groupby(["arch_name", "method", "HL", "seed"])["r"]
            .mean().reset_index())
curve = (per_seed.groupby(["arch_name", "method", "HL"])["r"]
         .agg(mean="mean", sd="std", n="count").reset_index())
curve["se"] = curve["sd"] / np.sqrt(curve["n"])

# --- bottom row data: paired delta vs uniform baseline ---
key = ["arch_name", "eval_H", "eval_L", "seed", "task"]
base = df[df["method"] == "uni"].set_index(key)["r"]
deltas = {}
for arch in ARCH_ORDER:
    rows = []
    for m in METHOD_ORDER[1:]:
        mm = df[(df["arch_name"] == arch) & (df["method"] == m)].set_index(key)["r"]
        j = pd.concat({"m": mm, "b": base[base.index.get_level_values(0) == arch]},
                      axis=1).dropna()
        d = j["m"] - j["b"]
        rows.append((m, d.mean(), d.std() / np.sqrt(len(d))))
    deltas[arch] = rows

fanda = subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=False)

for col, arch in enumerate(ARCH_ORDER):
    # ---- top: reward vs depth ----
    fanda.select(col)
    ax = fanda.ax
    ax.axvline(BASE_HL, color="#cccccc", lw=0.8, ls=":", zorder=1)
    for method in METHOD_ORDER:
        sub = curve[(curve["arch_name"] == arch) & (curve["method"] == method)].sort_values("HL")
        if sub.empty:
            continue
        c = METHOD_COLOR[method]
        x, y, se = sub["HL"].to_numpy(), sub["mean"].to_numpy(), sub["se"].to_numpy()
        ax.plot(x, y, color=c, lw=1.5, marker="o", ms=3, zorder=3,
                ls=METHOD_STYLE.get(method, "-"))
        ax.fill_between(x, y - se, y + se, color=c, alpha=0.18, lw=0)
    annotate_axis(fanda, xlabel="Recursion depth (H·L)",
                  ylabel="Episode Reward ($\\times 10^3$)" if col == 0 else "",
                  title=arch, labelsize=FONT_PT)
    decorate_axis(fanda, ticklabelsize=FONT_PT,
                  spines=["top", "right", "bottom", "left"])
    ax.set_xticks(sorted(curve["HL"].unique()))

    # ---- bottom: paired delta vs baseline ----
    fanda.select(2 + col)
    ax = fanda.ax
    ax.axhline(0.0, color="#888888", lw=0.9, ls="--", zorder=1)
    rows = deltas[arch]
    xs = np.arange(len(rows))
    for xi, (m, mu, se) in zip(xs, rows):
        ax.bar(xi, mu, width=0.68, color=METHOD_COLOR[m], zorder=2)
        ax.errorbar(xi, mu, yerr=se, color="#222222", lw=1.0, capsize=2.5, zorder=3)
    annotate_axis(fanda, xlabel="",
                  ylabel="$\\Delta$ vs uniform ($\\times 10^3$)" if col == 0 else "",
                  title="", labelsize=FONT_PT)
    decorate_axis(fanda, ticklabelsize=FONT_PT,
                  spines=["top", "right", "bottom", "left"])
    ax.set_xticks(xs)
    ax.set_xticklabels([m for m, _, _ in rows], rotation=30, ha="right",
                       fontsize=FONT_PT - 1)

plt.tight_layout()

handles = [
    Line2D([], [], color=METHOD_COLOR[m], lw=1.6, marker="o", ms=3,
           ls=METHOD_STYLE.get(m, "-"), label=METHOD_NAME[m])
    for m in METHOD_ORDER
]
fanda.fig.legend(handles, [h.get_label() for h in handles],
                 loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                 fontsize=FONT_PT - 1, fancybox=True, handlelength=1.8,
                 columnspacing=1.2, handletextpad=0.5)

save_fig(fanda, name="images/ood_cycles_methods", format="svg")
close_fig(fanda)
