"""F2: compute-matched Newt MLP (xl_dynamics_mlp=True) across latent dims.

Single panel, latent dim as hue (Rocket). This is the "wide MLP" control for F1:
does simply enlarging the MLP dynamics recover what the recursive models get?

Groups: abl_latent_dim_xl/newt_xl_<ld>ld.
"""

import numpy as np
import matplotlib.pyplot as plt

from fanda.wandb_client import fetch_wandb
from fanda.palettes import blue_rocket
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

# --- figure / font sizing (see newt_s_latent_dim.py) ---------------------------
FIG_WIDTH_IN = 6
FIG_HEIGHT_IN = 3
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

LD_ORDER = [16, 128, 384, 512]

try:
    df = fetch_wandb(
        ENTITY, PROJECT,
        filters={"group": {"$regex": r"^abl_latent_dim_xl/newt_xl_"}},
    )
except ValueError as e:
    print(f"[abl_latent_dim_xl] {e}; nothing to plot yet.")
    raise SystemExit(0)

df = df.copy()
df["latent"] = df["group"].str.extract(r"_(\d+)ld").astype(int)
df["legend"] = df["latent"].astype(int).astype(str) + "ld"

labels = [f"{d}ld" for d in LD_ORDER if d in set(df["latent"])]
palette = dict(zip(labels, blue_rocket(len(labels))))

fanda = (
    lineplot(
        df=df, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        x="eval/step", y="eval/episode_reward", hue="legend",
        hue_order=labels, palette=palette, errorbar="sd", err_kws={"alpha": 0.15},
    )
    .pipe(annotate_axis, xlabel="Training Steps", ylabel="Episode Reward", labelsize=FONT_PT)
    .pipe(decorate_axis, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
    .pipe(add_legend, labels=labels, palette=palette, fontsize=FONT_PT,
          loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=1)
    .pipe(lambda f: [plt.tight_layout(), f][1])
    .pipe(save_fig, name="images/abl_latent_dim_xl", format="svg")
    .pipe(close_fig)
)
