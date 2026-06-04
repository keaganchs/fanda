import matplotlib.pyplot as plt

import fanda
from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import lineplot, add_legend, annotate_axis, decorate_axis
from fanda.utils import save_fig, close_fig

plt.style.use('neurips')

df = (
    fetch_wandb("entity", "project", filters={
        "state": "finished",
        "created_at": {"$gte": "2025-01-01"},
    })
    .pipe(transforms.exponential_moving_average, column="loss", alpha=0.7)
    .pipe(transforms.remove_outliers, column="loss")
)
fanda = (
    lineplot(
        df=df,
        x="_step", 
        y="loss", 
        hue="network",
    )
    .pipe(
        annotate_axis, 
        xlabel="Number of Steps",
        ylabel="Loss",
    )
    .pipe(decorate_axis)
    .pipe(add_legend, column="algorithm")
    .pipe(save_fig, name="algorithm_comparison")
    .pipe(close_fig)
)