"""Recursion-budget scaling for SimpleTRM at latent_dim=384.

Budget = total inner core applications H*L. The abl_h_vs_l family gives two
budget levels, each with three (H,L) splits, and the splits pair up one-to-one
by their H:L ratio:
      H:L = 4:1        H:L = 1:1        H:L = 1:4
  4:  4h1l             2h2l             1h4l
  16: 8h2l             4h4l             2h8l
Each ratio is one colour; the budget is the line style (solid = 4, dashed = 16),
so the six runs read as three matched pairs.

CAVEATS (this is *not yet* a clean scaling curve; regenerate when the budget-16
runs finish):
  1. Training duration is unequal -- the budget-4 runs reached ~24M steps, the
     budget-16 runs are still going and currently stop near ~2-3M. The x-axis is
     therefore left at natural extent; do not read the right-hand end of the
     solid curves against the truncated dashed ones as a budget effect.
  2. L_layers differs by config (2 for budget 4, 1 for budget 16). The flag was
     added after these runs started, so confirm the actual architecture from the
     first ~30 lines of each run's log once they finish -- the two budgets should
     be made identical apart from H, L before any scaling claim.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fanda.wandb_client import fetch_wandb
from fanda.palettes import blue_rocket
from fanda.visualizations import annotate_axis, decorate_axis, lineplot
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

# --- figure / font sizing (see newt_s_latent_dim.py) ---------------------------
FIG_WIDTH_IN = 3.084  # sized so the axes box is 2.161x1.662in (1x1 panel), matching simple_v_newt_ld.py
FIG_HEIGHT_IN = 3.087
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

# group -> (H:L ratio label, total budget)
GROUP_INFO = {
    "abl_h_vs_l/smp_384ld_4h1l": ("H:L = 4:1", 4),
    "abl_h_vs_l/smp_384ld_8h2l": ("H:L = 4:1", 16),
    "abl_h_vs_l/smp_384ld_2h2l": ("H:L = 1:1", 4),
    "abl_h_vs_l/smp_384ld_4h4l": ("H:L = 1:1", 16),
    "abl_h_vs_l/smp_384ld_1h4l": ("H:L = 1:4", 4),
    "abl_h_vs_l/smp_384ld_2h8l": ("H:L = 1:4", 16),
}
RATIOS = ["H:L = 4:1", "H:L = 1:1", "H:L = 1:4"]     # colour, ordered outer -> inner heavy
BUDGETS = [4, 16]                                     # line style
RATIO_COLOR = dict(zip(RATIOS, blue_rocket(len(RATIOS))))
BUDGET_DASH = {4: "", 16: (4, 1.5)}                   # solid vs dashed

df = fetch_wandb(
    ENTITY, PROJECT, filters={"group": {"$in": list(GROUP_INFO)}},
).copy()
df["ratio"] = df["group"].map(lambda g: GROUP_INFO[g][0])
df["budget"] = df["group"].map(lambda g: GROUP_INFO[g][1])
# One hue key per run so each of the six draws its own line; colour/style come
# from the ratio/budget maps below.
df["legend"] = df["ratio"] + "  |  " + df["budget"].astype(str)

order, palette, dashes = [], {}, {}
for r in RATIOS:
    for b in BUDGETS:
        key = f"{r}  |  {b}"
        order.append(key)
        palette[key] = RATIO_COLOR[r]
        dashes[key] = BUDGET_DASH[b]
present = [k for k in order if k in set(df["legend"])]

fanda = (
    lineplot(
        df=df,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        x="eval/step",
        y="eval/episode_reward",
        hue="legend",
        hue_order=present,
        palette=palette,
        style="legend",
        style_order=present,
        dashes=dashes,
        errorbar="sd",
        err_kws={"alpha": 0.15},
        legend=False,
    )
    .pipe(annotate_axis, xlabel="Training Steps", ylabel="Episode Reward", labelsize=FONT_PT)
    .pipe(decorate_axis, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
)
# Two-part key: colour = H:L split ratio, line style = total budget H*L.
handles = [Line2D([], [], color=RATIO_COLOR[r], lw=1.6, label=r) for r in RATIOS]
handles += [
    Line2D([], [], color="#444444", dashes=BUDGET_DASH[b] or (None, None),
           lw=1.6, label=f"H·L = {b}")
    for b in BUDGETS
]
fanda.ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    fontsize=FONT_PT - 1,
    frameon=False,
    handlelength=1.2,
    columnspacing=0.8,
    handletextpad=0.5,
    # Architecture stated once as the title -- the two-part key (ratio colour x
    # budget style) is all SimpleTRM, so per-entry prefixes would be redundant.
    title="SimpleTRM",
    title_fontsize=FONT_PT - 1,
)

plt.tight_layout()
save_fig(fanda, name="images/abl_recursion_budget", format="svg")
close_fig(fanda)
