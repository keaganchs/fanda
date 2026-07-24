"""F6: FiLM task-conditioning of the dynamics {off, on} x recursion depth {1h1l, 8h4l}.

One panel per recursion depth; FiLM on/off is the hue (Rocket). Tests whether
feature-wise task modulation helps more at shallow or deep recursion.

Groups: abl_film/smp_384ld_<depth>_{film,nofilm}.
"""

import numpy as np
import matplotlib.pyplot as plt

from fanda.wandb_client import fetch_wandb
from fanda.palettes import blue_rocket
from fanda.visualizations import annotate_axis, decorate_axis, add_legend, subplots, add_lineplot
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

# --- figure / font sizing (see newt_s_latent_dim.py) ---------------------------
# FIG_WIDTH_IN is the width of ONE panel; the figure is FIG_WIDTH_IN * n_panels wide.
FIG_WIDTH_IN = 4
FIG_HEIGHT_IN = 3
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

DEPTH_ORDER = ["1h1l", "2h2l", "4h3l", "8h4l"]
FILM_ORDER = ["nofilm", "film"]
FILM_LABEL = {"nofilm": "no FiLM", "film": "FiLM"}
FILM_LABELS = [FILM_LABEL[f] for f in FILM_ORDER]
PALETTE = dict(zip(FILM_LABELS, blue_rocket(len(FILM_LABELS))))

try:
    df = fetch_wandb(
        ENTITY, PROJECT,
        filters={"group": {"$regex": r"^abl_film/smp_\d+ld_\dh\dl_(no)?film$"}},
    )
except ValueError as e:
    print(f"[abl_film] {e}; nothing to plot yet.")
    raise SystemExit(0)

df = df.copy()
df["depth"] = df["group"].str.extract(r"_(\dh\dl)_")
df["film"] = df["group"].str.extract(r"_((?:no)?film)$")
df["legend"] = df["film"].map(FILM_LABEL)

present = [d for d in DEPTH_ORDER if d in set(df["depth"])]
n = len(present)

fanda = subplots(1, n, figsize=(FIG_WIDTH_IN * n, FIG_HEIGHT_IN))
for i, depth in enumerate(present):
    sub = df[df["depth"] == depth]
    labels = [l for l in FILM_LABELS if l in set(sub["legend"])]
    fanda.select(i)
    add_lineplot(
        fanda, df=sub, x="eval/step", y="eval/episode_reward",
        hue="legend", hue_order=labels, palette=PALETTE,
        errorbar="sd", err_kws={"alpha": 0.15},
    )
    annotate_axis(
        fanda, xlabel="Training Steps",
        ylabel="Episode Reward" if i == 0 else "",
        title=depth, labelsize=FONT_PT,
    )
    decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
    add_legend(
        fanda, labels=labels, palette=PALETTE, fontsize=FONT_PT,
        loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=1,
    )

plt.tight_layout()
save_fig(fanda, name="images/abl_film", format="svg")
close_fig(fanda)
