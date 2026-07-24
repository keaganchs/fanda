"""Latent-dim sweep for the Newt MLP baselines: Newt S vs Newt S+XLd.

Two panels side by side, one per model size, sharing the latent-dim hue (the
thesis blue ramp, ordered 16 -> 512) and a single figure-level legend.

Groups:
  Newt S : dmc-newt-s-16ld, dmc-newt-S-{128ld,512ld,baseline}  (baseline = 384ld)
  Newt S+XLd: abl_latent_dim_xl/newt_xl_{16,128,384,512}ld
"""

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

# Font sizing: the figure is saved at FIG_WIDTH_IN wide and included in the paper
# at DISPLAY_WIDTH_IN (full \linewidth), so LaTeX scales it (and its text) by
# DISPLAY_WIDTH_IN / FIG_WIDTH_IN. To render as TARGET_PT on the page, the
# matplotlib font must be TARGET_PT / scale.
FIG_WIDTH_IN = 5.278  # sized so each axes box is 2.161x1.662in (2x1 panels), matching simple_v_newt_ld.py
FIG_HEIGHT_IN = 2.541
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth = \linewidth = 14.6979cm
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

LABELS = ["16ld", "128ld", "384ld (default)", "512ld"]
PALETTE = dict(zip(LABELS, blue_rocket(len(LABELS))))

# (panel title, group regex, group -> latent-dim label)
PANELS = [
    (
        "Newt S",
        r"^dmc-newt-s-16ld$|^dmc-newt-S-(128ld|512ld|baseline)$",
        lambda g: {"baseline": "384ld (default)"}.get(
            g.split("-")[-1], g.split("-")[-1]
        ),
    ),
    (
        "Newt S+XLd",
        r"^abl_latent_dim_xl/newt_xl_(16|128|384|512)ld$",
        lambda g: {"384ld": "384ld (default)"}.get(
            g.split("_")[-1], g.split("_")[-1]
        ),
    ),
]

fanda = subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True)
for i, (title, regex, label_of) in enumerate(PANELS):
    df = fetch_wandb(ENTITY, PROJECT, filters={"group": {"$regex": regex}}).copy()
    df["legend"] = df["group"].apply(label_of)
    present = [l for l in LABELS if l in set(df["legend"])]

    fanda.select(i)
    add_lineplot(
        fanda, df=df, x="eval/step", y="eval/episode_reward",
        hue="legend", hue_order=present, palette=PALETTE,
        errorbar="sd", err_kws={"alpha": 0.15}, legend=False,
    )
    annotate_axis(
        fanda,
        xlabel="Training Steps",
        ylabel="Episode Reward" if i == 0 else "",
        title=title,
        labelsize=FONT_PT,
    )
    decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

plt.tight_layout()

handles = [Line2D([], [], color=PALETTE[l], linewidth=1.8) for l in LABELS]
fanda.fig.legend(
    handles, LABELS,
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
    fontsize=FONT_PT, fancybox=True,
    handlelength=1.2, columnspacing=1.0, handletextpad=0.5,
)

save_fig(fanda, name="images/newt_s_latent_dim", format="svg")
close_fig(fanda)
