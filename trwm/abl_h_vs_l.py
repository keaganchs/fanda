"""F4: high vs. low recursion split (H,L) in {(4,1),(2,2),(1,4)} at fixed budget.

Single panel, (H,L) split as hue (Rocket). All cells run the same total recursion
budget; this asks whether depth is better spent on outer (H) or inner (L) cycles.

Groups: abl_h_vs_l/smp_384ld_<H>h<L>l.
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
FIG_WIDTH_IN = 2.967  # sized so the axes box is 2.161x1.662in (1x1 panel), matching simple_v_newt_ld.py
FIG_HEIGHT_IN = 2.348
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

# ordered high -> low outer recursion
HL_ORDER = ["4h1l", "2h2l", "1h4l"]
HL_LABEL = {"4h1l": "SimpleTRM: H=4, L=1", "2h2l": "SimpleTRM: H=2, L=2",
            "1h4l": "SimpleTRM: H=1, L=4"}

try:
    df = fetch_wandb(
        ENTITY, PROJECT,
        filters={"group": {"$regex": r"^abl_h_vs_l/smp_\d+ld_\dh\dl"}},
    )
except ValueError as e:
    print(f"[abl_h_vs_l] {e}; nothing to plot yet.")
    raise SystemExit(0)

df = df.copy()
df["hl"] = df["group"].str.extract(r"_(\dh\dl)$")
df["legend"] = df["hl"].map(HL_LABEL)

labels = [HL_LABEL[k] for k in HL_ORDER if k in set(df["hl"])]
palette = dict(zip(labels, blue_rocket(len(labels))))

fanda = (
    lineplot(
        df=df, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        x="eval/step", y="eval/episode_reward", hue="legend",
        hue_order=labels, palette=palette, errorbar="sd", err_kws={"alpha": 0.15},
        legend=False,
    )
    .pipe(annotate_axis, xlabel="Training Steps", ylabel="Episode Reward", labelsize=FONT_PT)
    .pipe(decorate_axis, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
)
plt.tight_layout()
# Architecture-prefixed labels are too long for a multi-column key on a
# half-width panel, so stack them in one column above the axes.
fanda.fig.legend(
    handles=[Line2D([], [], color=palette[l], lw=1.6, label=l) for l in labels],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=1,
    fontsize=FONT_PT,
    fancybox=True,
    handlelength=1.6,
    columnspacing=1.0,
    handletextpad=0.5,
)

save_fig(fanda, name="images/abl_h_vs_l", format="svg")
close_fig(fanda)
