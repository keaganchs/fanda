"""F5: latent regularisation {SimNorm, SIGReg, none} at latent_dim=384.

2x2 multiplot: episode reward plus the three training losses the regulariser
acts on. The SimpleTRM cells sweep wm_regularization_type at the anchor config
(4h3l); the two Newt S+XLd runs are the compute-matched MLP baselines, one under
SimNorm and one under SIGReg.

The 16ld cells are deliberately excluded -- none of them learned.

SimpleTRM series use the thesis blue ramp; the Newt S+XLd baselines are dashed
neutral gray so ours-vs-baseline reads at a glance (and in grayscale print).

Groups:
  abl_regularization/smp_384ld_4h3l_{simnorm,sigreg,none}
  abl_latent_dim_xl/newt_xl_384ld        (Newt S+XLd, SimNorm)
  newt_xl_sigreg_384ld_0p01              (Newt S+XLd, SIGReg, coef 0.01)
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
FIG_WIDTH_IN = 5.6  # full \linewidth figure (2x2 panels)
FIG_HEIGHT_IN = 4.6
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

blues = blue_rocket(3)
# group -> (label, colour, dash)
SERIES_OF = {
    "abl_regularization/smp_384ld_4h3l_simnorm": ("SimpleTRM, SimNorm", blues[0], ""),
    "abl_regularization/smp_384ld_4h3l_sigreg": ("SimpleTRM, SIGReg", blues[1], ""),
    "abl_regularization/smp_384ld_4h3l_none": ("SimpleTRM, None", blues[2], ""),
    "abl_latent_dim_xl/newt_xl_384ld": ("Newt S+XLd, SimNorm", "#555555", (4, 1.5)),
    "newt_xl_sigreg_384ld_0p01": ("Newt S+XLd, SIGReg", "#888888", (4, 1.5)),
    # In progress (finishes ~2026-07-22); the panel fills in once eval data lands.
    "newt_xl_sigreg_384ld_0p01/newt_xl_none_384ld": ("Newt S+XLd, None", "#bbbbbb", (4, 1.5)),
}
SERIES = [v[0] for v in SERIES_OF.values()]
PALETTE = {v[0]: v[1] for v in SERIES_OF.values()}
DASHES = {v[0]: v[2] for v in SERIES_OF.values()}

# Eval and train metrics land on separate history rows, so each panel carries
# its own x column: (x, y, title). The train losses are logged every few hundred
# steps and are far too noisy to read raw, so they get a per-run EMA; the eval
# reward is already sparse and is left alone.
# Log-y for the two losses that span orders of magnitude across regularisers
# (SimNorm sits ~10x-30x below SIGReg/None and is unreadable on a linear axis).
# SIGReg loss stays linear: it is exactly 0 for every non-SIGReg run, and log-y
# would silently drop those three series instead of showing them at zero.
EMA_ALPHA = 0.05
X_MAX = 1.5e7
PANELS = [
    # (x, y, title, smooth, log_y)
    ("eval/step", "eval/episode_reward", "Episode Reward", False, False),
    ("train/step", "train/total_loss", "Total Loss", True, True),
    ("train/step", "train/sigreg_loss", "SIGReg Loss", True, False),
    ("train/step", "train/consistency_loss", "Consistency Loss", True, True),
]

df = fetch_wandb(
    ENTITY, PROJECT,
    filters={"group": {"$in": list(SERIES_OF)}},
)
df = df.copy()
df["legend"] = df["group"].map(lambda g: SERIES_OF[g][0])

fanda = subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
for i, (x, y, title, smooth, log_y) in enumerate(PANELS):
    sub = df[[x, y, "legend", "run_id"]].dropna()
    if smooth:
        sub = transforms.exponential_moving_average(
            sub.copy(), column=y, alpha=EMA_ALPHA, groupby="run_id"
        )
    present = [s for s in SERIES if s in set(sub["legend"])]
    fanda.select(i)
    add_lineplot(
        fanda, df=sub, x=x, y=y,
        hue="legend", hue_order=present, palette=PALETTE,
        style="legend", style_order=present, dashes=DASHES,
        errorbar="sd", err_kws={"alpha": 0.15}, legend=False,
    )
    annotate_axis(
        fanda,
        xlabel="Training Steps" if i >= 2 else "",
        ylabel="",
        title=title,
        labelsize=FONT_PT,
    )
    decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
    # Panels are not shared, so clamp each one. The Newt S+XLd baselines run out to
    # ~3e7; this crops them to the range the SimpleTRM cells actually cover.
    fanda.ax.set_xlim(0, X_MAX)
    if log_y:
        fanda.ax.set_yscale("log")

plt.tight_layout()

handles = [
    Line2D([], [], color=PALETTE[s], dashes=DASHES[s] or (None, None), linewidth=1.8)
    for s in SERIES
]
# 6 entries at ncol=3 (2 rows: SimpleTRM row, Newt S+XLd row). Font dropped 1pt
# and spacing tightened so the key stays within \linewidth (was ~16pt overfull).
fanda.fig.legend(
    handles, SERIES,
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
    fontsize=FONT_PT - 1, fancybox=True,
    handlelength=1.4, columnspacing=0.8, handletextpad=0.4,
)

save_fig(fanda, name="images/abl_regularization", format="svg")
close_fig(fanda)
