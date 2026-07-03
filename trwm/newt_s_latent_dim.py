import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend
from fanda.utils import show_fig, save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600

np.random.seed(42)

# Font sizing: the figure is saved at FIG_WIDTH_IN wide and included in the paper
# at DISPLAY_WIDTH_IN (full \linewidth), so LaTeX scales it (and its text) by
# DISPLAY_WIDTH_IN / FIG_WIDTH_IN. To render as TARGET_PT on the page, the
# matplotlib font must be TARGET_PT / scale.
FIG_WIDTH_IN = 6
FIG_HEIGHT_IN = 3
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth = \linewidth = 14.6979cm
TARGET_PT = 11
# FONT_PT = TARGET_PT * FIG_WIDTH_IN / DISPLAY_WIDTH_IN  # ~19.46pt
FONT_PT = 11

df = (
    fetch_wandb(
        "trm-dynamics",
        "TRM Dynamics",
        filters={
            "group": {
                "$regex": "^dmc-newt-s-16ld$|^dmc-newt-S-(128ld|512ld|baseline)$"
            }
        },
    )
    # .pipe(transforms.truncate, column="eval/episode_reward", groupby="group")
    # .pipe(
    #     transforms.remove_outliers,
    #     column="eval/episode_reward",
    #     lower_quantile=0.05,
    #     upper_quantile=0.95,
    # )
    # .pipe(transforms.normalize, column="eval/episode_reward", groupby="group")
)

# Add legend column for the last part of the group name after the last dash
df["legend"] = df["group"].apply(lambda x: x.split("-")[-1])
df["legend"] = df["legend"].replace({"baseline": "384ld (default)"})


labels = ["16ld", "128ld", "384ld (default)", "512ld"]
palette = dict(zip(labels, sns.color_palette("rocket", len(labels))))

fanda = (
    lineplot(
        df=df,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        x="eval/step",
        y="eval/episode_reward",
        hue="legend",
        palette=palette,
        errorbar="sd",
        err_kws={"alpha": 0.15},
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
    .pipe(
        add_legend,
        labels=labels,
        palette=palette,
        fontsize=FONT_PT,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
    )
    .pipe(lambda f: [plt.tight_layout(), f][1])
    .pipe(show_fig)
    .pipe(save_fig, name="images/newt_s_latent_dim", format="svg")
    .pipe(close_fig)
)
