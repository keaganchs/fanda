"""SimpleTRM vs SRM at two recursion depths, vs the Newt baselines (latent_dim=384).

Single panel comparing the two recursive architectures each at 1h1l (no
recursion) and 4h3l (full recursion), against Newt S and Newt S+XLd.

Encoding:
  colour     = recursion depth (1h1l / 4h3l)
  line style = architecture (solid = SimpleTRM, dashed = SRM)
  grey dotted= the Newt baselines
So the SimpleTRM/SRM gap at a depth reads within one colour, and each
architecture's 1h1l->4h3l gain reads across colours.

NOTE: the two families are different SRM/SimpleTRM variants (see srm_cycles.py /
simple_v_newt_ld.py); this is a reward comparison, not a matched-architecture one.
The x-axis is clipped to just past the longest 4h3l run.

Groups:
  paper_simple_384ld_{1h1l,4h3l}        (SimpleTRM)
  abl_cycles_srm/srm_384ld_{1h1l,4h3l}  (SRM)
  abl_latent_dim_xl/newt_xl_384ld, dmc-newt-S-baseline  (baselines)
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
FIG_WIDTH_IN = 2.967  # sized so the axes box is 2.161x1.662in (1x1 panel), matching simple_v_newt_ld.py
FIG_HEIGHT_IN = 2.348
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

blues = blue_rocket(4)
DEPTHS = ["1h1l", "4h3l"]
DEPTH_COLOR = {"1h1l": blues[0], "4h3l": blues[2]}      # colour = depth (1h1l darker)
ARCHS = ["SimpleTRM", "SRM"]
ARCH_DASH = {"SimpleTRM": "", "SRM": (4, 1.5)}          # style = architecture
BASELINES = {"Newt S+XLd": "#555555", "Newt S": "#aaaaaa"}
BASE_DASH = (1, 1.5)                                    # dotted, distinct from SRM's dash

# group -> (architecture, depth) for the recursive cells
RECUR = {
    "paper_simple_384ld_1h1l": ("SimpleTRM", "1h1l"),
    "paper_simple_384ld_4h3l": ("SimpleTRM", "4h3l"),
    "abl_cycles_srm/srm_384ld_1h1l": ("SRM", "1h1l"),
    "abl_cycles_srm/srm_384ld_4h3l": ("SRM", "4h3l"),
}
BASELINE_GROUPS = {
    "abl_latent_dim_xl/newt_xl_384ld": "Newt S+XLd",
    "dmc-newt-S-baseline": "Newt S",
}

order, palette, dashes = [], {}, {}
for a in ARCHS:
    for d in DEPTHS:
        key = f"{a} {d}"
        order.append(key)
        palette[key], dashes[key] = DEPTH_COLOR[d], ARCH_DASH[a]
for name, col in BASELINES.items():
    order.append(name)
    palette[name], dashes[name] = col, BASE_DASH

df = fetch_wandb(
    ENTITY, PROJECT,
    filters={"group": {"$in": list(RECUR) + list(BASELINE_GROUPS)}},
).copy()
df["legend"] = df["group"].map(
    lambda g: f"{RECUR[g][0]} {RECUR[g][1]}" if g in RECUR else BASELINE_GROUPS[g]
)
present = [k for k in order if k in set(df["legend"])]

# Clip just past the longest 4h3l run.
four = df[df["legend"].isin(["SimpleTRM 4h3l", "SRM 4h3l"])]["eval/step"].dropna()
x_max = float(four.max()) * 1.02

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
fanda.ax.set_xlim(0, x_max)

plt.tight_layout()

# Three-part key: colour = depth, style = architecture, grey = baselines.
handles = [Line2D([], [], color=DEPTH_COLOR[d], lw=1.6, label=d) for d in DEPTHS]
handles += [
    Line2D([], [], color="#444444", dashes=ARCH_DASH[a] or (None, None), lw=1.6, label=a)
    for a in ARCHS
]
handles += [
    Line2D([], [], color=BASELINES[b], dashes=BASE_DASH, lw=1.6, label=b) for b in BASELINES
]
fanda.fig.legend(
    handles, [h.get_label() for h in handles],
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
    fontsize=FONT_PT - 1, fancybox=True,
    handlelength=1.5, columnspacing=1.0, handletextpad=0.5,
)

save_fig(fanda, name="images/simple_v_srm_depth", format="svg")
close_fig(fanda)
