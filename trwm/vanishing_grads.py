import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend, subplots, add_lineplot
from fanda.utils import show_fig, save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"

np.random.seed(42)


df = (
    fetch_wandb("trm-dynamics", "TRM Dynamics", filters={"group": {"$regex": "simple_s_384ld_2h6l_skip|simple_s_384ld_2h6l_noskip"}})
    # .pipe(transforms.truncate, column="eval/episode_reward", groupby="group")
    # .pipe(
    #     transforms.remove_outliers,
    #     column="eval/episode_reward",
    #     lower_quantile=0.05,
    #     upper_quantile=0.95,
    # )
    # .pipe(transforms.normalize, column="eval/episode_reward", groupby="group")
)

# Copy the df to solve the fragmentation warning
df = df.copy()

# Add legend column for the last part of the group name after the last dash
df["legend"] = df["group"].apply(lambda x: x.split("_")[-1])

# Melt the dataframe to put all z_0 ... z_5 values into a single column
value_vars = [f"train/dyn_step_grad_norm_z_{i}" for i in range(6)]
df_melt = df.melt(id_vars=["_step", "legend"], value_vars=value_vars, var_name="z_step", value_name="grad_norm")

# Clean up z_step column to just contain "z_0", "z_1", etc.
df_melt["z_step"] = df_melt["z_step"].apply(lambda x: x[-3:])

fanda = (
    subplots(1, 2, figsize=(14, 5))
    .pipe(add_lineplot, df=df_melt, x="z_step", y="grad_norm", hue="legend", errorbar="sd", err_style="bars", marker="o")
    .pipe(annotate_axis, xlabel="Recursion Cycle", ylabel="Grad Norm", title="Gradient Norms over Recursion Cycles")
    .pipe(decorate_axis)
    .pipe(lambda f: [f.ax.invert_xaxis(), f][1])
    .select(1)
    .pipe(add_lineplot, df=df, x="_step", y="train/enc_grad_norm", hue="legend", errorbar="sd")
    .pipe(annotate_axis, xlabel="Steps", ylabel="Encoder Grad Norm", title="Encoder Gradient Norm")
    .pipe(decorate_axis)
    .pipe(lambda f: [plt.tight_layout(), f][1])
    .pipe(save_fig, name="images/vanishing_grads", format="png")
    .pipe(close_fig)
)

fanda_reward = (
    lineplot(
        df=df,
        figsize=(10, 6),
        x="eval/step",
        y="eval/episode_reward",
        hue="legend",
        errorbar="sd"
    )
    .pipe(annotate_axis, xlabel="Training Steps", ylabel="Normalized Episode Reward", title="Episode Reward")
    .pipe(decorate_axis)
    .pipe(lambda f: [plt.tight_layout(), f][1])
    .pipe(save_fig, name="images/vanishing_grads_reward", format="png")
    .pipe(close_fig)
)

