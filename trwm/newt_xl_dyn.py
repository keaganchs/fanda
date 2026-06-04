import wandb
import numpy as np
import matplotlib.pyplot as plt

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend
from fanda.utils import show_fig, save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"

np.random.seed(42)


df = (
    fetch_wandb("trm-dynamics", "TRM Dynamics", filters={"group": {"$regex": ".*-newt-S-(128ld|512ld|baseline)$"}})
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

fanda = (
    lineplot(
        df=df,
        x="eval/step",
        y="eval/episode_reward",
        hue="group",
        palette="deep",
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
        labels=df["legend"].unique(),
        fontsize="xx-large",
    )
    .pipe(show_fig)
    .pipe(save_fig, name="images/noisy_training", format="png")
    .pipe(close_fig)
)
