"""SimpleTRM latent-structure ablation: CLIP task embeddings x initial y carry.

2x2 factorial at the anchor config (SimpleTRM, latent_dim=384, 4h3l):
  - task embeddings on/off (use_task_embedding), i.e. whether the recursive core
    ever sees the frozen CLIP text embedding of the task;
  - initial high carry y copied from the encoded world-model latent, or drawn
    from trunc_normal noise (rrm_random_y_init).

Colour encodes the CLIP factor (dark = with CLIP, light = without) and dash
style encodes the carry-init factor (solid = latent init, dashed = random), so
each factor is readable on its own and the design survives grayscale print.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fanda.wandb_client import fetch_wandb
from fanda.palettes import blue_rocket
from fanda.visualizations import annotate_axis, decorate_axis, lineplot
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

# --- figure / font sizing (see newt_s_latent_dim.py) ---------------------------
FIG_WIDTH_IN = 2.967  # sized so each axes box is 2.161x1.662in (1x1 panels), matching simple_v_newt_ld.py
FIG_HEIGHT_IN = 2.373
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

blues = blue_rocket(4)
CLIP_COLOR = blues[2]     # with CLIP task embeddings (light)
NOCLIP_COLOR = blues[0]   # without (dark)

# group suffix -> (label, colour, dash). Labels carry the architecture so the key
# is self-contained.
SERIES = {
    "": ("SimpleTRM: CLIP, latent init", CLIP_COLOR, ""),
    "_randinit": ("SimpleTRM: CLIP, random init", CLIP_COLOR, (4, 1.5)),
    "_notask": ("SimpleTRM: no CLIP, latent init", NOCLIP_COLOR, ""),
    "_notask_randinit": ("SimpleTRM: no CLIP, random init", NOCLIP_COLOR, (4, 1.5)),
}
BASE = "paper_simple_384ld_4h3l"
XLD_GROUP = "abl_latent_dim_xl/newt_xl_384ld"   # Newt S+XLd baseline (dashed grey)
BASELINE = "Newt S+XLd"

LABELS = [v[0] for v in SERIES.values()] + [BASELINE]
PALETTE = {v[0]: v[1] for v in SERIES.values()} | {BASELINE: "#555555"}
DASHES = {v[0]: v[2] for v in SERIES.values()} | {BASELINE: (4, 1.5)}

df = fetch_wandb(
    ENTITY, PROJECT,
    filters={"group": {"$regex": rf"^{BASE}(_notask)?(_randinit)?$|^{XLD_GROUP}$"}},
)
df = df.copy()
df["legend"] = df["group"].apply(
    lambda g: BASELINE if g == XLD_GROUP else SERIES[g[len(BASE):]][0]
)

fanda = (
    lineplot(
        df=df,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        x="eval/step",
        y="eval/episode_reward",
        hue="legend",
        hue_order=LABELS,
        style="legend",
        style_order=LABELS,
        dashes=DASHES,
        palette=PALETTE,
        errorbar="sd",
        err_kws={"alpha": 0.15},
        legend=False,
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
    # Clip to where the SimpleTRM cells stop; the Newt S+XLd baseline runs to
    # ~2.7e7 but its long solo tail would compress the comparison region.
    .pipe(lambda f: [f.ax.set_xlim(0, 1.1e7), f][1])
    .pipe(lambda f: [plt.tight_layout(), f][1])
    # Framed figure-level legend above the axes (matches abl_regularization).
    .pipe(
        lambda f: [
            f.fig.legend(
                handles=[
                    Line2D([], [], color=PALETTE[l], dashes=DASHES[l] or (None, None),
                           lw=1.5, label=l)
                    for l in LABELS
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=1,
                fontsize=FONT_PT - 2,
                fancybox=True,
                handlelength=1.6,
                columnspacing=1.0,
                handletextpad=0.5,
            ),
            f,
        ][1]
    )
    .pipe(save_fig, name="images/simple_task_carry", format="svg")
    .pipe(close_fig)
)
