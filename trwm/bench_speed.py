"""Inference-speed benchmark (eager, one RTX 5080).

Left: dynamics next() latency vs recursion depth (H*L inner cycles) for the trained
SimpleTRM / SRM cores at batch 4096, with the Newt MLP baselines as horizontal
references -- isolates the per-model cost and shows the linear-in-depth scaling.
Right: full MPPI plan() latency per control step (all 21 envs) for the four models at
their operating depth -- the acting cost a deployment sees (labelled with control Hz).

Data: analysis/results/speed.csv in the newt repo (analysis/bench_speed.py).
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

FIG_WIDTH_IN = 5.6     # 1x2 panels, ~2.16in axes box each
FIG_HEIGHT_IN = 2.55
FONT_PT = 11

CSV = Path("/home/holmes/projects/thesis/vla/newt/analysis/results/speed.csv")

blues = blue_rocket(3)
SIMPLE_C, SRM_C = blues[0], blues[2]
NEWT_S_C, NEWT_XL_C = "#aaaaaa", "#555555"
COLOR = {"SimpleTRM 4h3l": SIMPLE_C, "SRM 4h3l": SRM_C,
         "Newt S": NEWT_S_C, "Newt S+XLd": NEWT_XL_C}

df = pd.read_csv(CSV)
dyn = df[df["kind"] == "dynamics_next"].copy()
plan = df[df["kind"] == "plan"].copy()

fanda = subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

# --- left: dynamics next() latency vs depth ---------------------------------------
ax = fanda.axs.flat[0]
for model, color, dash in [("SimpleTRM 4h3l", SIMPLE_C, ""),
                           ("SRM 4h3l", SRM_C, (4, 1.5))]:
    sub = dyn[dyn["model"] == model].sort_values("HL")
    ls = (0, dash) if dash else "-"
    ax.plot(sub["HL"], sub["latency_ms_mean"], ls=ls, color=color, lw=1.6,
            marker="o", ms=3, zorder=3)
    ax.fill_between(sub["HL"], sub["latency_ms_mean"] - sub["latency_ms_std"],
                    sub["latency_ms_mean"] + sub["latency_ms_std"],
                    color=color, alpha=0.15, lw=0)
for model, color in [("Newt S", NEWT_S_C), ("Newt S+XLd", NEWT_XL_C)]:
    v = dyn[dyn["model"] == model]["latency_ms_mean"]
    if len(v):
        ax.axhline(float(v.iloc[0]), color=color, ls=(0, (4, 1.5)), lw=1.4, zorder=1)
fanda.select(0)
annotate_axis(fanda, xlabel="Recursion depth (H·L)", ylabel="Dynamics latency (ms)",
              labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
ax.set_xticks(sorted(dyn["HL"].dropna().unique()))

# --- right: full plan() latency per control step ----------------------------------
ax = fanda.axs.flat[1]
order = ["Newt S", "Newt S+XLd", "SimpleTRM 4h3l", "SRM 4h3l"]
order = [m for m in order if m in set(plan["model"])]
xs = np.arange(len(order))
lat = [float(plan[plan["model"] == m]["latency_ms_mean"].iloc[0]) for m in order]
ax.bar(xs, lat, color=[COLOR[m] for m in order], width=0.68, zorder=3)
for x, m, l in zip(xs, order, lat):
    ax.text(x, l, f"{1e3 / l:.0f} Hz", ha="center", va="bottom", fontsize=FONT_PT - 3)
fanda.select(1)
annotate_axis(fanda, xlabel="", ylabel="Plan() latency (ms/step)", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
ax.set_xticks(xs)
ax.set_xticklabels(["Newt S", "Newt\nS+XLd", "Simple\nTRM", "SRM"], fontsize=FONT_PT - 2)
ax.set_ylim(0, max(lat) * 1.16)

plt.tight_layout()

handles = [
    Line2D([], [], color=SIMPLE_C, lw=1.6, label="SimpleTRM 4h3l"),
    Line2D([], [], color=SRM_C, dashes=(4, 1.5), lw=1.6, label="SRM 4h3l"),
    Line2D([], [], color=NEWT_XL_C, dashes=(4, 1.5), lw=1.4, label="Newt S+XLd"),
    Line2D([], [], color=NEWT_S_C, dashes=(4, 1.5), lw=1.4, label="Newt S"),
]
fanda.fig.legend(handles, [h.get_label() for h in handles],
                 loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
                 fontsize=FONT_PT - 2, fancybox=True, handlelength=1.5,
                 columnspacing=1.0, handletextpad=0.5)

save_fig(fanda, name="images/bench_speed", format="svg")
close_fig(fanda)
