"""SRM vs Newt baselines across latent dims.

2x2 multiplot, one panel per latent dim (16/128/384/512). Each panel compares
the SRM dynamics at the anchor recursion depth (abl_latent_dim/srm_<ld>ld_4h3l)
against the two Newt MLP baselines: Newt S and Newt S with the XL dynamics MLP (Newt S+XLd)
(abl_latent_dim_xl/newt_xl_<ld>ld).

SRM uses the thesis blue ramp; baselines are dashed neutral gray so
ours-vs-baseline reads at a glance (and in grayscale print).

Note: these SRM runs use the SwiGLU context net with L_layers=2. The
abl_cycles_srm family (see srm_cycles.py) is a *different* variant -- MLP
context net, L_layers=1 -- and is deliberately kept in its own figure.
"""

import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fanda.wandb_client import fetch_wandb
from fanda.palettes import blue_rocket
from fanda.visualizations import annotate_axis, decorate_axis, subplots, add_lineplot
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

# --- figure / font sizing (see newt_s_latent_dim.py) ---------------------------
FIG_WIDTH_IN = 5.6  # full \linewidth figure (2x2 panels)
FIG_HEIGHT_IN = 4.6
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11
X_MAX = 1.5e7

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

LD_ORDER = [16, 128, 384, 512]

blues = blue_rocket(4)
SERIES = ["SRM 4h3l", "Newt S+XLd", "Newt S"]
PALETTE = {
    "SRM 4h3l": blues[0],
    "Newt S+XLd": "#555555",
    "Newt S": "#aaaaaa",
}
DASHES = {
    "SRM 4h3l": "",
    "Newt S+XLd": (4, 1.5),
    "Newt S": (4, 1.5),
}

df = fetch_wandb(
    ENTITY, PROJECT,
    filters={"group": {"$regex": (
        r"^abl_latent_dim/srm_\d+ld_4h3l$"
        r"|^abl_latent_dim_xl/newt_xl_\d+ld$"
        r"|^dmc-newt-s-16ld$"
        r"|^dmc-newt-S-(128ld|512ld|baseline)$"
    )}},
)
df = df.copy()


def series_of(group: str) -> str:
    if group.startswith("abl_latent_dim/srm_"):
        return "SRM 4h3l+SwiGLU Skip"
    if group.startswith("abl_latent_dim_xl/"):
        return "Newt S+XLd"
    return "Newt S"


def ld_of(group: str) -> int:
    if group == "dmc-newt-S-baseline":
        return 384
    return int(re.search(r"(\d+)ld", group).group(1))


df["legend"] = df["group"].apply(series_of)
df["latent"] = df["group"].apply(ld_of)

fanda = subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True, sharey=True)
for i, ld in enumerate(LD_ORDER):
    sub = df[df["latent"] == ld]
    present = [s for s in SERIES if s in set(sub["legend"])]
    fanda.select(i)
    add_lineplot(
        fanda, df=sub, x="eval/step", y="eval/episode_reward",
        hue="legend", hue_order=present, palette=PALETTE,
        style="legend", style_order=present, dashes=DASHES,
        errorbar="sd", err_kws={"alpha": 0.15}, legend=False,
    )
    annotate_axis(
        fanda,
        xlabel="Training Steps" if i >= 2 else "",
        ylabel="Episode Reward" if i % 2 == 0 else "",
        title=f"{ld}ld", labelsize=FONT_PT,
    )
    decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

fanda.axs.flat[0].set_xlim(0, X_MAX)  # sharex: applies to all panels

plt.tight_layout()

handles = [
    Line2D([], [], color=PALETTE[s], dashes=DASHES[s] or (None, None), linewidth=1.8)
    for s in SERIES
]
fanda.fig.legend(
    handles, SERIES,
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
    fontsize=FONT_PT, fancybox=True,
    handlelength=1.6, columnspacing=1.0, handletextpad=0.5,
)

save_fig(fanda, name="images/srm_v_newt_ld", format="svg")
close_fig(fanda)
