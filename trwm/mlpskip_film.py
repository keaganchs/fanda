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
    fetch_wandb("trm-dynamics", "TRM Dynamics", filters={"group": {"$regex": ".*-newt-S-baseline$|mlpskip_film|mlpskip_nofilm|dmcontrol-newt-S-XL-dynamics$"}})
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
df["legend"] = df["legend"].replace({"dmc-newt-S-baseline": "NewtS (350k)"})
df["legend"] = df["legend"].replace({"mlpskip_film": "MLPSkip+FiLM (800k)"})
df["legend"] = df["legend"].replace({"mlpskip_nofilm": "MLPSkip (800k)"})
df["legend"] = df["legend"].replace({"dmcontrol-newt-S-XL-dynamics": "XL Dynamics (930k)"})

_blues = blue_rocket(4)
palette = {
    "NewtS (350k)": _blues[0],
    "MLPSkip+FiLM (800k)": _blues[1],
    "MLPSkip (800k)": _blues[2],
    "XL Dynamics (930k)": _blues[3],
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
    .pipe(save_fig, name="images/mlpskip_film", format="png")
    .pipe(close_fig)
)


