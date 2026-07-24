"""FiLM-conditioning architecture sweep at latent_dim in {16, 384}.

The abl_film_arch family FiLM-conditions the dynamics of three architectures --
SimpleTRM, SRM, and the Newt S+XLd MLP -- and asks (a) whether FiLM helps at all
(vs the plain non-FiLM Newt S / Newt S+XLd baselines) and (b) whether the FiLM
conditioner should see the action as well as the task (task+action) or the task
alone (task-only).

Encoding (per panel):
  colour     = FiLM'd architecture (SimpleTRM / SRM / Newt+XLd)
  line style = FiLM input: solid = task+action, dashed = task-only
  grey dotted= the non-FiLM baselines Newt S and Newt S+XLd
One panel per latent dim.

CAVEATS -- horizons are very unequal and some runs are still going:
  - latent 16: the FiLM runs reach ~11M steps (3 seeds).
  - latent 384: the FiLM'd Newt reaches ~7M, but the recursive FiLM runs
    (SimpleTRM/SRM) currently stop near ~2M and are still training. Each panel's
    x-axis is clipped to the extent of its FiLM runs; the 384 recursive curves
    are early previews. All L_layers=4 for the recursive cells; confirm the exact
    architecture from the run logs (first ~30 lines) once the runs finish.
"""

import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fanda.wandb_client import fetch_wandb
from fanda.palettes import blue_rocket
from fanda.visualizations import annotate_axis, decorate_axis, subplots, add_lineplot
from fanda.utils import save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.dpi"] = 600
np.random.seed(42)

# --- figure / font sizing (see newt_s_latent_dim.py) ---------------------------
FIG_WIDTH_IN = 5.320  # 1x2 panels; refit to a 2.161x1.662in axes box after edits
FIG_HEIGHT_IN = 2.541
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

LATENTS = [16, 384]
X_MAX = {16: 1.15e7, 384: 7.5e6}   # clip each panel to its FiLM runs' extent

ARCHS = ["SimpleTRM", "SRM", "Newt+XLd"]        # colour
ARCH_COLOR = dict(zip(ARCHS, blue_rocket(len(ARCHS))))
COND_DASH = {"task+action": "", "task-only": (4, 1.5)}   # solid / dashed
BASELINES = {"Newt S+XLd": "#555555", "Newt S": "#aaaaaa"}
BASE_DASH = (1, 1.5)                              # dotted, distinct from task-only

ARCH_OF = {"smp": "SimpleTRM", "srm": "SRM", "newt": "Newt+XLd"}


def film_series(group_tail: str):
    """abl_film_arch tail -> (label, colour, dash) or None if not a FiLM run."""
    m = re.match(r"(smp|srm|newt)_.*_(taskact|taskonly)$", group_tail)
    if not m:
        return None
    arch = ARCH_OF[m.group(1)]
    cond = "task+action" if m.group(2) == "taskact" else "task-only"
    label = f"{arch} ({cond})"
    return label, ARCH_COLOR[arch], COND_DASH[cond]


# Baseline groups per latent dim (non-FiLM Newt references).
BASELINE_GROUPS = {
    "dmc-newt-s-16ld": ("Newt S", 16),
    "dmc-newt-S-baseline": ("Newt S", 384),
    "abl_latent_dim_xl/newt_xl_16ld": ("Newt S+XLd", 16),
    "abl_latent_dim_xl/newt_xl_384ld": ("Newt S+XLd", 384),
}

df = fetch_wandb(
    ENTITY, PROJECT,
    filters={"group": {"$regex": (
        r"^abl_film_arch/"
        r"|^dmc-newt-s-16ld$|^dmc-newt-S-baseline$"
        r"|^abl_latent_dim_xl/newt_xl_(16|384)ld$"
    )}},
).copy()

# Build (label, colour, dash, latent) for every run.
palette, dashes, latent_of_label = {}, {}, {}
labels = []


def classify(group: str):
    if group.startswith("abl_film_arch/"):
        tail = group.split("/", 1)[1]
        fs = film_series(tail)
        if fs is None:
            return None
        label, col, dsh = fs
        latent = int(re.search(r"_(\d+)ld", tail).group(1))
        palette[label] = col
        dashes[label] = dsh
        return label, latent
    if group in BASELINE_GROUPS:
        label, latent = BASELINE_GROUPS[group]
        palette[label] = BASELINES[label]
        dashes[label] = BASE_DASH
        return label, latent
    return None


df["_cls"] = df["group"].apply(classify)
df = df[df["_cls"].notna()]
df["legend"] = df["_cls"].apply(lambda t: t[0])
df["latent"] = df["_cls"].apply(lambda t: t[1])

# Draw order: FiLM archs (by arch, cond), then baselines.
ORDER = [f"{a} ({c})" for a in ARCHS for c in ("task+action", "task-only")] + list(BASELINES)

fanda = subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True)
for i, ld in enumerate(LATENTS):
    sub = df[(df["latent"] == ld) & (df["eval/step"] <= X_MAX[ld])]
    present = [s for s in ORDER if s in set(sub["legend"])]
    fanda.select(i)
    add_lineplot(
        fanda, df=sub, x="eval/step", y="eval/episode_reward",
        hue="legend", hue_order=present, palette=palette,
        style="legend", style_order=present, dashes=dashes,
        errorbar="sd", err_kws={"alpha": 0.15}, legend=False,
    )
    annotate_axis(
        fanda,
        xlabel="Training Steps",
        ylabel="Episode Reward" if i == 0 else "",
        title=f"{ld}ld", labelsize=FONT_PT,
    )
    decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
    fanda.ax.set_xlim(0, X_MAX[ld])

plt.tight_layout()

# Three-part key: colour = architecture, style = FiLM input, grey = baselines.
handles = [Line2D([], [], color=ARCH_COLOR[a], lw=1.6, label=a) for a in ARCHS]
handles += [
    Line2D([], [], color="#444444", dashes=COND_DASH[c] or (None, None), lw=1.6,
           label=f"FiLM: {c}")
    for c in ("task+action", "task-only")
]
handles += [
    Line2D([], [], color=BASELINES[b], dashes=BASE_DASH, lw=1.6, label=b)
    for b in BASELINES
]
fanda.fig.legend(
    handles, [h.get_label() for h in handles],
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
    fontsize=FONT_PT - 1, fancybox=True,
    handlelength=1.6, columnspacing=1.0, handletextpad=0.5,
)

save_fig(fanda, name="images/abl_film_arch", format="svg")
close_fig(fanda)
