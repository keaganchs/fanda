import wandb
import numpy as np
import matplotlib.pyplot as plt

from fanda.wandb_client import fetch_wandb
from fanda.palettes import blue_rocket
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend
from fanda.utils import show_fig, save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"

np.random.seed(42)


df = (
    fetch_wandb("trm-dynamics", "TRM Dynamics", filters={"group": {"$regex": ".*-newt-S-baseline$|dmc_trmd_mlp_16hs16ld_no_rec"}})
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
df["legend"] = df["group"]
df["legend"] = df["legend"].replace({"dmc-newt-S-baseline": "Baseline"})
df["legend"] = df["legend"].replace({"dmc_trmd_mlp_16hs16ld_no_rec": "Tokenized TRM, no recursion"})


_blues = blue_rocket(4)
palette = {
    "Baseline": _blues[0],
    "Tokenized TRM, no recursion": _blues[2],
}
labels = list(palette.keys())

fanda = (
    lineplot(
        df=df,
        figsize=(10.24, 10.24),
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
        ylabel="Normalized Episode Reward (Smoothed)",
        labelsize="xx-large",
    )
    .pipe(
        decorate_axis,
        ticklabelsize="xx-large",
    )
    .pipe(
        add_legend,
        labels=labels,
        palette=palette,
        fontsize="xx-large",
    )
    .pipe(lambda f: [plt.tight_layout(), f][1])
    .pipe(show_fig)
    .pipe(save_fig, name="images/trm_16hs16ld", format="png")
    .pipe(close_fig)
)


