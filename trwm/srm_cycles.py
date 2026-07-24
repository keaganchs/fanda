"""SRM recursion-depth sweep at latent_dim=384.

Single panel: abl_cycles_srm/srm_384ld_{1h1l,2h2l,4h3l}, with the two
Newt baselines as dashed references. 1h1l is the no-recursion control,
so the spread across this panel is the effect of recursion depth alone --
every cell shares one recursively applied core (794,880 dynamics params), so
depth costs compute, not capacity.

NOTE: this family is a *different SRM variant* from the latent-dim sweep in
srm_v_newt_ld.py, which uses a SwiGLU context net with L_layers=2 (1,974,528
dynamics params) rather than the MLP context net with L_layers=1 used here.
The two are not comparable cell-for-cell; see the parameter table.
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
FIG_WIDTH_IN = 3.073  # sized so each axes box is 2.161x1.662in (1x1 panels), matching simple_v_newt_ld.py
FIG_HEIGHT_IN = 2.348
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11
X_MAX = 1.5e7

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

BASELINE_XL = "Newt S+XLd"
BASELINE_S = "Newt S"
# 8h4l excluded: it only reached <3M steps, too short to be informative.
CYCLES = ["1h1l", "2h2l", "4h3l"]
SERIES = CYCLES + [BASELINE_XL, BASELINE_S]

PALETTE = {
    **dict(zip(CYCLES, blue_rocket(len(CYCLES)))),
    BASELINE_XL: "#555555",
    BASELINE_S: "#aaaaaa",
}
DASHES = {**{c: "" for c in CYCLES}, BASELINE_XL: (4, 1.5), BASELINE_S: (4, 1.5)}

df = fetch_wandb(
    ENTITY, PROJECT,
    filters={"group": {"$regex": (
        r"^abl_cycles_srm/srm_384ld_(1h1l|2h2l|4h3l)$"
        r"|^abl_latent_dim_xl/newt_xl_384ld$"
        r"|^dmc-newt-S-baseline$"
    )}},
).copy()


def series_of(group: str) -> str:
    if group.startswith("abl_cycles_srm/"):
        return group.rsplit("_", 1)[-1]      # 1h1l / 2h2l / 4h3l
    if group.startswith("abl_latent_dim_xl/"):
        return BASELINE_XL
    return BASELINE_S


df["legend"] = df["group"].apply(series_of)
present = [s for s in SERIES if s in set(df["legend"])]

# Display labels prefix the recursion cells with the architecture (baselines keep
# their own names).
DISPLAY = {**{c: f"SRM: {c}" for c in CYCLES},
           BASELINE_XL: BASELINE_XL, BASELINE_S: BASELINE_S}

fanda = (
    lineplot(
        df=df,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        x="eval/step",
        y="eval/episode_reward",
        hue="legend",
        hue_order=present,
        palette=PALETTE,
        style="legend",
        style_order=present,
        dashes=DASHES,
        errorbar="sd",
        err_kws={"alpha": 0.15},
        legend=False,
    )
    .pipe(
        annotate_axis,
        xlabel="Training Steps",
        ylabel="Episode Reward",
        labelsize=FONT_PT,
    )
    .pipe(
        decorate_axis,
        ticklabelsize=FONT_PT,
        spines=["top", "right", "bottom", "left"],
    )
)
fanda.ax.set_xlim(0, X_MAX)

plt.tight_layout()

# Framed figure-level legend above the axes (matches abl_regularization). 6 entries
# on a half-width panel -> ncol=2 keeps the box inside the 2.8in axes.
fanda.fig.legend(
    handles=[
        Line2D([], [], color=PALETTE[s], dashes=DASHES[s] or (None, None),
               lw=1.6, label=DISPLAY[s])
        for s in present
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
    fontsize=FONT_PT,
    fancybox=True,
    handlelength=1.6,
    columnspacing=1.0,
    handletextpad=0.5,
)

save_fig(fanda, name="images/srm_cycles", format="svg")
close_fig(fanda)
