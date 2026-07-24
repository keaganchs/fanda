"""D4: mask the input x during the y-update (rrm_mask_x_for_y_update) {True, False}.

Single panel, masking on/off as hue (Rocket).

Groups: abl_mask_x/smp_384ld_4h3l_maskx_{true,false}.
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

MASK_ORDER = ["true", "false"]
MASK_LABEL = {"true": "mask x (on)", "false": "keep x (off)"}

try:
    df = fetch_wandb(
        ENTITY, PROJECT,
        filters={"group": {"$regex": r"^abl_mask_x/smp_\d+ld_4h3l_maskx_(true|false)$"}},
    )
except ValueError as e:
    print(f"[abl_mask_x] {e}; nothing to plot yet.")
    raise SystemExit(0)

df = df.copy()
df["maskx"] = df["group"].str.extract(r"_maskx_(true|false)$")
df["legend"] = df["maskx"].map(MASK_LABEL)

labels = [MASK_LABEL[m] for m in MASK_ORDER if m in set(df["maskx"])]
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
    .pipe(save_fig, name="images/abl_mask_x", format="svg")
    .pipe(close_fig)
)
