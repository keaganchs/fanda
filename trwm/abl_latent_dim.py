"""F1 headline: latent-dim sweep across dynamics architectures.

One panel per architecture (Newt MLP / SimpleTRM / TRM / SRM); within a panel the
latent dim is the hue (Rocket). Only architectures that already have runs are
drawn, so the script is safe to run on preliminary data.

Groups: abl_latent_dim/<arch>_<ld>ld[_4h3l].  The SimpleTRM 4h3l latent sweep also
exists under the legacy `paper_simple_<ld>ld_4h3l` groups (pre-reorg), which are
folded in as the `smp` architecture until the abl_ SimpleTRM runs finish.
"""

import re

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

ARCH_ORDER = ["newt", "smp", "trm", "srm"]
ARCH_LABEL = {"newt": "Newt (MLP)", "smp": "SimpleTRM", "trm": "TRM", "srm": "SRM"}
LD_ORDER = [16, 128, 384, 512]
LD_LABELS = [f"{d}ld" for d in LD_ORDER]
PALETTE = dict(zip(LD_LABELS, blue_rocket(len(LD_LABELS))))

try:
    df = fetch_wandb(
        ENTITY, PROJECT,
        filters={"group": {"$regex": r"^abl_latent_dim/(newt|smp|trm|srm)_\d+ld|^paper_simple_\d+ld_4h3l$"}},
    )
except ValueError as e:
    print(f"[abl_latent_dim] {e}; nothing to plot yet.")
    raise SystemExit(0)

df = df.copy()


def arch_of(group: str) -> str:
    if group.startswith("paper_simple"):
        return "smp"
    return re.search(r"^abl_latent_dim/([a-z]+)_", group).group(1)


df["arch"] = df["group"].apply(arch_of)
df["latent"] = df["group"].str.extract(r"_(\d+)ld").astype(int)
df["legend"] = df["latent"].astype(int).astype(str) + "ld"

present_archs = [a for a in ARCH_ORDER if a in set(df["arch"])]
n = len(present_archs)

fanda = subplots(1, n, figsize=(FIG_WIDTH_IN * n, FIG_HEIGHT_IN))
for i, arch in enumerate(present_archs):
    sub = df[df["arch"] == arch]
    labels = [l for l in LD_LABELS if l in set(sub["legend"])]
    fanda.select(i)
    add_lineplot(
        fanda, df=sub, x="eval/step", y="eval/episode_reward",
        hue="legend", hue_order=labels, palette=PALETTE,
        errorbar="sd", err_kws={"alpha": 0.15},
    )
    annotate_axis(
        fanda, xlabel="Training Steps",
        ylabel="Episode Reward" if i == 0 else "",
        title=ARCH_LABEL[arch], labelsize=FONT_PT,
    )
    decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
    add_legend(
        fanda, labels=labels, palette=PALETTE, fontsize=FONT_PT,
        loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=1,
    )

plt.tight_layout()
save_fig(fanda, name="images/abl_latent_dim", format="svg")
close_fig(fanda)
