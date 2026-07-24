"""S2: SRM backprop-through-recursion truncation length, latent_dim=384.

Main figure (images/abl_srm_truncation.svg), the 4h3l cells with T in {1,2,3}:
  left  -- eval/episode_reward (sparse, plotted raw)
  right -- train/episode_reward under a per-run EMA. The smoothed training
           curve is where the T=2 advantage is legible, being continuous rather
           than sampled once per eval.

Appendix figure (images/abl_srm_truncation_3h12l.svg), the 3h12l cells with
T in {3,6,12}: a much wider truncation range (25/50/100% of the 12 inner
cycles), but only 3 seeds and ~3M steps -- a quarter of the 4h3l horizon -- and
a different SRM variant (L_layers=1, MLP context net, 794,880 dynamics params
vs 1,974,528 for the 4h3l cells). Corroborating evidence only.

Groups: abl_srm_truncation/srm_384ld_4h3l_trunc{1,2,3}
        abl_srm_truncation/srm_384ld_3h12l_{3,6,12}t
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.palettes import blue_rocket
from fanda.visualizations import annotate_axis, decorate_axis, subplots, add_lineplot
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

# --- figure / font sizing (see newt_s_latent_dim.py) ---------------------------
FIG_WIDTH_IN = 5.384  # sized so each axes box is 2.161x1.662in (2x1 panels), matching simple_v_newt_ld.py
FIG_HEIGHT_IN = 2.550
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11
EMA_ALPHA = 0.02

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

df = fetch_wandb(
    ENTITY, PROJECT,
    filters={"group": {"$regex": r"^abl_srm_truncation/"}},
).copy()


def make_figure(cells, labels, name, x_max, title_prefix):
    """cells: {group: label}. Two panels: eval reward, smoothed train reward."""
    sub_all = df[df["group"].isin(cells)].copy()
    sub_all["legend"] = sub_all["group"].map(cells)
    palette = dict(zip(labels, blue_rocket(len(labels))))
    present = [l for l in labels if l in set(sub_all["legend"])]

    panels = [
        ("eval/step", "eval/episode_reward", "Eval", False),
        ("train/step", "train/episode_reward", "Train (smoothed)", True),
    ]
    fanda = subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True)
    for i, (x, y, title, smooth) in enumerate(panels):
        d = sub_all[[x, y, "legend", "run_id"]].dropna()
        if smooth:
            d = transforms.exponential_moving_average(
                d.copy(), column=y, alpha=EMA_ALPHA, groupby="run_id"
            )
        fanda.select(i)
        add_lineplot(
            fanda, df=d, x=x, y=y,
            hue="legend", hue_order=present, palette=palette,
            errorbar="sd", err_kws={"alpha": 0.15}, legend=False,
        )
        annotate_axis(
            fanda,
            xlabel="Training Steps",
            ylabel="Episode Reward" if i == 0 else "",
            title=title, labelsize=FONT_PT,
        )
        decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
        fanda.ax.set_xlim(0, x_max)

    plt.tight_layout()
    # Architecture moves into the legend labels (e.g. "SRM: T = 1"); titles stay
    # plain ("Eval" / "Train (smoothed)").
    handles = [Line2D([], [], color=palette[l], linewidth=1.8) for l in present]
    fanda.fig.legend(
        handles, [f"{title_prefix}: {l}" for l in present],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(present),
        fontsize=FONT_PT, fancybox=True,
        handlelength=1.4, columnspacing=1.0, handletextpad=0.5,
    )
    save_fig(fanda, name=name, format="svg")
    close_fig(fanda)


# --- main: 4h3l, T in {1,2,3} --------------------------------------------------
make_figure(
    cells={
        "abl_srm_truncation/srm_384ld_4h3l_trunc1": "T = 1",
        "abl_srm_truncation/srm_384ld_4h3l_trunc2": "T = 2",
        "abl_srm_truncation/srm_384ld_4h3l_trunc3": "T = 3",
    },
    labels=["T = 1", "T = 2", "T = 3"],
    name="images/abl_srm_truncation",
    x_max=1.2e7,
    title_prefix="SRM",
)

# --- appendix: 3h12l, T in {3,6,12} (3 seeds, ~3M steps) -----------------------
# make_figure(
#     cells={
#         "abl_srm_truncation/srm_384ld_3h12l_3t": "T = 3",
#         "abl_srm_truncation/srm_384ld_3h12l_6t": "T = 6",
#         "abl_srm_truncation/srm_384ld_3h12l_12t": "T = 12",
#     },
#     labels=["T = 3", "T = 6", "T = 12"],
#     name="images/abl_srm_truncation_3h12l",
#     x_max=3.0e6,
#     title_prefix="SRM 3h12l, 384ld",
# )
