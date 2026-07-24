"""F7: deep-intermediate-supervision (DIS) loss {off, linear, cosine} x latent_dim {16, 384}.

One panel per latent dim; the DIS schedule is the hue (Rocket). Tests whether
supervising intermediate recursion states (and how its weight is scheduled) helps.

Groups: abl_dis_loss/smp_<ld>ld_4h3l_dis_{off,linear,cosine}.
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

LD_ORDER = [16, 384]
DIS_ORDER = ["off", "linear", "cosine"]
DIS_LABEL = {"off": "DIS off", "linear": "DIS (linear)", "cosine": "DIS (cosine)"}
DIS_LABELS = [DIS_LABEL[d] for d in DIS_ORDER]
PALETTE = dict(zip(DIS_LABELS, blue_rocket(len(DIS_LABELS))))

try:
    df = fetch_wandb(
        ENTITY, PROJECT,
        filters={"group": {"$regex": r"^abl_dis_loss/smp_\d+ld_4h3l_dis_(off|linear|cosine)$"}},
    )
except ValueError as e:
    print(f"[abl_dis_loss] {e}; nothing to plot yet.")
    raise SystemExit(0)

df = df.copy()
df["latent"] = df["group"].str.extract(r"_(\d+)ld").astype(int)
df["dis"] = df["group"].str.extract(r"_dis_(off|linear|cosine)$")
df["legend"] = df["dis"].map(DIS_LABEL)

present = [ld for ld in LD_ORDER if ld in set(df["latent"])]
n = len(present)

fanda = subplots(1, n, figsize=(FIG_WIDTH_IN * n, FIG_HEIGHT_IN))
for i, ld in enumerate(present):
    sub = df[df["latent"] == ld]
    labels = [l for l in DIS_LABELS if l in set(sub["legend"])]
    fanda.select(i)
    add_lineplot(
        fanda, df=sub, x="eval/step", y="eval/episode_reward",
        hue="legend", hue_order=labels, palette=PALETTE,
        errorbar="sd", err_kws={"alpha": 0.15},
    )
    annotate_axis(
        fanda, xlabel="Training Steps",
        ylabel="Episode Reward" if i == 0 else "",
        title=f"latent_dim = {ld}", labelsize=FONT_PT,
    )
    decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
    add_legend(
        fanda, labels=labels, palette=PALETTE, fontsize=FONT_PT,
        loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=1,
    )

plt.tight_layout()
save_fig(fanda, name="images/abl_dis_loss", format="svg")
close_fig(fanda)
