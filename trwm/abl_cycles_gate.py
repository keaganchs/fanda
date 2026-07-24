"""F3: skip-connection gate {off, additive, mlp, swiglu} x recursion depth.

One panel per recursion depth (1h1l / 2h2l / 4h3l / 8h4l); the gate variant is the
hue (Rocket). Shows whether a gated skip is what makes deeper recursion pay off.
Only depths that already have runs are drawn.

Groups: abl_cycles_gate/smp_384ld_<depth>_{off,additive,mlp,swiglu}.
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
GATE_ORDER = ["off", "additive", "mlp", "swiglu"]
GATE_LABEL = {"off": "no skip", "additive": "additive", "mlp": "MLP gate", "swiglu": "SwiGLU gate"}
GATE_LABELS = [GATE_LABEL[g] for g in GATE_ORDER]
PALETTE = dict(zip(GATE_LABELS, blue_rocket(len(GATE_LABELS))))

try:
    df = fetch_wandb(
        ENTITY, PROJECT,
        filters={"group": {"$regex": r"^abl_cycles_gate/smp_\d+ld_\dh\dl_(off|additive|mlp|swiglu)$"}},
    )
except ValueError as e:
    print(f"[abl_cycles_gate] {e}; nothing to plot yet.")
    raise SystemExit(0)

df = df.copy()
df["depth"] = df["group"].str.extract(r"_(\dh\dl)_")
df["gate"] = df["group"].str.extract(r"_(off|additive|mlp|swiglu)$")
df["legend"] = df["gate"].map(GATE_LABEL)

present = [d for d in DEPTH_ORDER if d in set(df["depth"])]
n = len(present)

fanda = subplots(1, n, figsize=(FIG_WIDTH_IN * n, FIG_HEIGHT_IN))
for i, depth in enumerate(present):
    sub = df[df["depth"] == depth]
    labels = [l for l in GATE_LABELS if l in set(sub["legend"])]
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
save_fig(fanda, name="images/abl_cycles_gate", format="svg")
close_fig(fanda)
