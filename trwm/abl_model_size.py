"""B2: Newt MLP baseline across model-size presets: S (default, 384ld) vs M vs L.

Single panel comparing episode reward across the preset sizes. The presets scale
the whole world model -- encoder / MLP / Q-ensemble widths and latent dim -- not
just the latent dim (S: 384ld; M, L: 512ld with progressively wider trunks and a
larger Q-ensemble). All use the plain MLP dynamics (use_trm_dynamics=None), a
fair "bigger baseline" control for the recursive-dynamics comparison.

CAVEAT: training horizons are unequal -- S reached ~30M steps, M ~14.5M, L
~11.3M. Curves are shown at natural extent; compare over the shared range rather
than by final value alone.

Groups: dmc-newt-S-baseline, abl_model_size/newt_{m,l}.
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
FIG_WIDTH_IN = 3.084  # sized so the axes box is 2.161x1.662in (1x1 panel), matching simple_v_newt_ld.py
FIG_HEIGHT_IN = 2.686
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

# group -> label, ordered small -> large (colour ramp follows size)
SIZE_OF = {
    "dmc-newt-S-baseline": "Newt S",
    "abl_model_size/newt_m": "Newt M",
    "abl_model_size/newt_l": "Newt L",
}
LABELS = ["Newt S", "Newt M", "Newt L"]
PALETTE = dict(zip(LABELS, blue_rocket(len(LABELS))))

df = fetch_wandb(
    ENTITY, PROJECT, filters={"group": {"$in": list(SIZE_OF)}},
).copy()
df["legend"] = df["group"].map(SIZE_OF)

fanda = (
    lineplot(
        df=df,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        x="eval/step",
        y="eval/episode_reward",
        hue="legend",
        hue_order=LABELS,
        palette=PALETTE,
        errorbar="sd",
        err_kws={"alpha": 0.15},
        legend=False,
    )
    .pipe(annotate_axis, xlabel="Training Steps", ylabel="Episode Reward", labelsize=FONT_PT)
    .pipe(decorate_axis, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
)
fanda.ax.legend(
    handles=[Line2D([], [], color=PALETTE[l], lw=1.6, label=l) for l in LABELS],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(LABELS),
    fontsize=FONT_PT,
    frameon=False,
    handlelength=1.2,
    columnspacing=0.8,
    handletextpad=0.5,
)

plt.tight_layout()
save_fig(fanda, name="images/abl_model_size", format="svg")
close_fig(fanda)
