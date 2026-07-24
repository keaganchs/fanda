"""S3: skip-connection gate and the vanishing gradient in SimpleTRM (384ld, 4h3l).

SimpleTRM detaches its first H-1 recursion cycles (warmup), so gradients only
flow through the final H-cycle: its L=3 inner z-updates (z_0 deepest -> z_2
nearest the loss) and the terminal high-carry update y. The gate types differ
only in the skip connection applied after each update:
  off       -- no skip (anchor)
  additive  -- residual add
  mlp       -- learned MLP skip
(The swiglu gate crashed early and is excluded.)

All gradnorm / margin / delta series below are already logged in W&B
(log_trm_gradnorms=True during training); nothing is recomputed here. Per-cycle
values are the mean over the last half of training, pooled across seeds; bands
are +/-1 SD over that pool.

Panels:
  1. Episode reward -- additive fails to learn; off and mlp do.
  2. Gradient norm over the gradient-carrying recursion cycles (z_0,z_1,z_2,y),
     log scale. off vanishes ~23x into the deepest cycle; the mlp skip flattens
     it to ~3x; additive is perfectly flat (no vanishing).
  3. Advantage margin A_i (mean) across all H*L cycles: the reduction in squared
     distance to the consistency target at each cycle, positive = useful. additive
     is negative and grows more negative every cycle -- it actively diverges from
     the target -- which is why it cannot learn despite preserving the gradient.
     (An alternative "fraction of non-improving cycles" panel is kept commented.)
  4. High-carry (y) state change per outer cycle, log scale -- a per-cycle view
     of the y carry to complement its single terminal gradnorm.
"""

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
FIG_WIDTH_IN = 5.874  # 2x2 panels; refit to a 2.161x1.662in axes box after edits
FIG_HEIGHT_IN = 4.990
DISPLAY_WIDTH_IN = 5.78853  # paper \textwidth
TARGET_PT = 11
FONT_PT = 11
X_MAX = 1.3e7   # common reward horizon
LATE_FRAC = 0.5  # per-cycle stats use the last half of training

ENTITY, PROJECT = "trm-dynamics", "TRM Dynamics"

GATES = ["off", "additive", "mlp"]
GATE_LABEL = {"off": "SimpleTRM: No skip", "additive": "SimpleTRM: Additive",
              "mlp": "SimpleTRM: MLP"}
GROUP_OF = {g: f"abl_gradnorm_gate/smp_384ld_4h3l_{g}" for g in GATES}
PALETTE = dict(zip(GATES, blue_rocket(len(GATES))))

P = "train/dyn_step_grad_norm_"
H, L = 4, 3

df = fetch_wandb(
    ENTITY, PROJECT, filters={"group": {"$in": list(GROUP_OF.values())}},
).copy()
inv = {v: k for k, v in GROUP_OF.items()}
df["gate"] = df["group"].map(inv)


def per_cycle(gate, cols):
    """Mean +/- SD of each column over the last LATE_FRAC of training (seeds pooled)."""
    d = df[df["gate"] == gate]
    late = d[d["train/step"] >= (1 - LATE_FRAC) * d["train/step"].max()]
    m = np.array([late[c].dropna().mean() if c in late else np.nan for c in cols])
    s = np.array([late[c].dropna().std() if c in late else np.nan for c in cols])
    return m, s


fanda = subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

# --- panel 1: episode reward ---------------------------------------------------
fanda.select(0)
rew = df[df["eval/step"] <= X_MAX][["eval/step", "eval/episode_reward", "gate"]].dropna()
add_lineplot(
    fanda, df=rew, x="eval/step", y="eval/episode_reward",
    hue="gate", hue_order=GATES, palette=PALETTE,
    errorbar="sd", err_kws={"alpha": 0.15}, legend=False,
)
annotate_axis(fanda, xlabel="Training Steps", ylabel="Episode Reward",
              title="Episode Reward", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
fanda.ax.set_xlim(0, X_MAX)

# --- panel 2: gradient norm over the gradient-carrying cycles (z_0..z_2, y) -----
fanda.select(1)
grad_cols = [f"{P}z_{i}" for i in range(L)] + [f"{P}y"]
xg = np.arange(len(grad_cols))
for gate in GATES:
    m, s = per_cycle(gate, grad_cols)
    fanda.ax.plot(xg, m, "-o", color=PALETTE[gate], lw=1.4, ms=3)
    fanda.ax.fill_between(xg, m - s, m + s, color=PALETTE[gate], alpha=0.12, lw=0)
fanda.ax.set_yscale("log")
fanda.ax.set_xticks(xg)
fanda.ax.set_xticklabels([r"$z_0$", r"$z_1$", r"$z_2$", r"$y$"])
annotate_axis(fanda, xlabel="Recursion cycle", ylabel="Gradient norm",
              title="Gradient Norm: z & y carries", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

# --- panel 3: advantage margin (mean) across all cycles ------------------------
# A_i = ||s_{i-1} - z*||^2 - ||s_i - z*||^2, the reduction in squared distance to
# the consistency target z* at cycle i: A_i > 0 means the cycle refined the latent
# toward the target (useful work), A_i < 0 means it moved away. symlog keeps the
# small (off) and large (mlp/additive) magnitudes and both signs readable; the
# dotted line marks A_i = 0 (the useful / harmful boundary).
fanda.select(2)
marg_cols = [f"{P}h{h}_l{i}_advantage_margin_mean" for h in range(H) for i in range(L)]
xm = np.arange(len(marg_cols))
for gate in GATES:
    m, _ = per_cycle(gate, marg_cols)
    fanda.ax.plot(xm, m, "-o", color=PALETTE[gate], lw=1.4, ms=2.5)
fanda.ax.axhline(0, color="#999999", lw=0.8, ls=":", zorder=1)
fanda.ax.set_yscale("symlog", linthresh=1.0)
annotate_axis(fanda, xlabel="Recursion cycle ($h\\cdot L + l$)",
              ylabel="Advantage margin $A_i$",
              title="Advantage Margin per Cycle", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

# --- (alt) panel 3: fraction of cycles with non-positive margin ("dead compute").
# Swap back by commenting out the block above and uncommenting this one.
# marg_cols = [f"{P}h{h}_l{i}_advantage_margin_frac_nonpositive"
#              for h in range(H) for i in range(L)]
# xm = np.arange(len(marg_cols))
# for gate in GATES:
#     m, _ = per_cycle(gate, marg_cols)
#     fanda.ax.plot(xm, m, "-o", color=PALETTE[gate], lw=1.4, ms=2.5)
# annotate_axis(fanda, xlabel="Recursion cycle ($h\\cdot L + l$)",
#               ylabel="Frac. non-positive margin",
#               title="Dead Compute per Cycle", labelsize=FONT_PT)
# decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
# fanda.ax.set_ylim(-0.05, 1.08)

# --- panel 4: high-carry (y) state change per outer cycle -----------------------
fanda.select(3)
yd_cols = [f"{P}y_{h}_delta" for h in range(H)]
xy = np.arange(len(yd_cols))
for gate in GATES:
    m, s = per_cycle(gate, yd_cols)
    fanda.ax.plot(xy, m, "-o", color=PALETTE[gate], lw=1.4, ms=3)
    fanda.ax.fill_between(xy, np.clip(m - s, 1e-9, None), m + s,
                          color=PALETTE[gate], alpha=0.12, lw=0)
fanda.ax.set_yscale("log")
fanda.ax.set_xticks(xy)
annotate_axis(fanda, xlabel="Outer cycle $h$", ylabel=r"$\|y_i - y_{i-1}\|$",
              title="High-carry Update per Cycle", labelsize=FONT_PT)
decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])

plt.tight_layout()

handles = [Line2D([], [], color=PALETTE[g], lw=1.8, label=GATE_LABEL[g]) for g in GATES]
fanda.fig.legend(
    handles, [GATE_LABEL[g] for g in GATES],
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
    fontsize=FONT_PT, fancybox=True,
    handlelength=1.6, columnspacing=1.0, handletextpad=0.5,
)

save_fig(fanda, name="images/abl_gradnorm_gate", format="svg")
close_fig(fanda)
