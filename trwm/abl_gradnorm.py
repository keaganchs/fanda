"""D1: gradient flow through the recursion (vanishing-gradient diagnostic).

For the deepest recursion depth available, plot -- as a function of the recursion
cycle -- the per-cycle gradient norm, the state delta ||z_i - z_{i-1}||, and the
cosine similarity between successive states, with the skip-connection gate as hue
(Rocket). This is the updated-style companion to vanishing_grads.py.

Metrics (auto-discovered from the fetched history):
  train/dyn_step_grad_norm_z_<i>          -- grad norm at inner cycle i  (i = 0..L-1)
  train/dyn_step_grad_norm_z_<i>_delta    -- ||z_i - z_{i-1}|| at inner cycle i
  train/dyn_step_grad_norm_y_<h>_cossim   -- cos(z_h, z_{h-1}) at high cycle h (0..H-1)

Groups: abl_gradnorm/smp_384ld_<depth>_gn_{off,additive,mlp,swiglu}. The legacy
gradnorm-logging runs `paper_simple_384ld_{1h1l,4h3l}` (gate=off) are folded in so
the figure renders on existing data.
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

GATE_ORDER = ["off", "additive", "mlp", "swiglu"]
GATE_LABEL = {"off": "no skip", "additive": "additive", "mlp": "MLP gate", "swiglu": "SwiGLU gate"}
GATE_LABELS = [GATE_LABEL[g] for g in GATE_ORDER]
PALETTE = dict(zip(GATE_LABELS, blue_rocket(len(GATE_LABELS))))

try:
    df = fetch_wandb(
        ENTITY, PROJECT,
        filters={"group": {"$regex": r"^abl_gradnorm/|^paper_simple_384ld_(1h1l|4h3l)$"}},
    )
except ValueError as e:
    print(f"[abl_gradnorm] {e}; nothing to plot yet.")
    raise SystemExit(0)

df = df.copy()


def gate_of(group: str) -> str:
    m = re.search(r"_gn_(off|additive|mlp|swiglu)$", group)
    return m.group(1) if m else "off"  # legacy anchor runs have no skip gate


df["depth"] = df["group"].str.extract(r"(\d+h\d+l)")
df["gate"] = df["group"].apply(gate_of)
df["legend"] = df["gate"].map(GATE_LABEL)


def cols(pattern: str):
    """Columns matching `pattern` (one \\d+ capture group) that carry data, sorted by index."""
    out = [(int(m.group(1)), c) for c in df.columns for m in [re.fullmatch(pattern, c)] if m]
    return [c for _, c in sorted(out) if df[c].notna().any()]


# Pick the deepest recursion depth that actually logged gradient norms.
z_cols_all = cols(r"train/dyn_step_grad_norm_z_(\d+)")
depths = [d for d in df["depth"].dropna().unique()]
if not depths or not z_cols_all:
    print("[abl_gradnorm] no per-cycle gradient columns found yet; nothing to plot.")
    raise SystemExit(0)
depth = max(depths, key=lambda d: sum(df.loc[df["depth"] == d, c].notna().any() for c in z_cols_all))
sub = df[df["depth"] == depth]
print(f"[abl_gradnorm] plotting depth={depth}")


def melt(pattern: str, value_name: str, cycle_name: str):
    vcols = [c for c in cols(pattern) if sub[c].notna().any()]
    if not vcols:
        return None
    m = sub.melt(id_vars=["_step", "legend"], value_vars=vcols,
                 var_name=cycle_name, value_name=value_name)
    m[cycle_name] = m[cycle_name].str.extract(r"_(\d+)(?:_delta|_cossim)?$").astype(int)
    return m.dropna(subset=[value_name])


# (metric dataframe, x-label, y-label, title)
panels = []
gn = melt(r"train/dyn_step_grad_norm_z_(\d+)", "grad_norm", "cycle")
if gn is not None:
    panels.append((gn, "cycle", "grad_norm", "Recursion Cycle", "Gradient Norm", "Grad Norm per Cycle"))
dl = melt(r"train/dyn_step_grad_norm_z_(\d+)_delta", "delta", "cycle")
if dl is not None:
    panels.append((dl, "cycle", "delta", "Recursion Cycle", r"$\|z_i - z_{i-1}\|$", "State Delta per Cycle"))
cs = melt(r"train/dyn_step_grad_norm_y_(\d+)_cossim", "cossim", "hcycle")
if cs is not None:
    panels.append((cs, "hcycle", "cossim", "High Cycle", r"$\cos(z_h, z_{h-1})$", "State Cos-Sim per High Cycle"))

n = len(panels)
fanda = subplots(1, n, figsize=(FIG_WIDTH_IN * n, FIG_HEIGHT_IN))
for i, (mdf, x, y, xlabel, ylabel, title) in enumerate(panels):
    labels = [l for l in GATE_LABELS if l in set(mdf["legend"])]
    fanda.select(i)
    add_lineplot(
        fanda, df=mdf, x=x, y=y, hue="legend", hue_order=labels, palette=PALETTE,
        errorbar="sd", err_style="bars", marker="o",
    )
    annotate_axis(fanda, xlabel=xlabel, ylabel=ylabel, title=title, labelsize=FONT_PT)
    decorate_axis(fanda, ticklabelsize=FONT_PT, spines=["top", "right", "bottom", "left"])
    add_legend(fanda, labels=labels, palette=PALETTE, fontsize=FONT_PT,
               loc="best", bbox_to_anchor=(0.02, 0.98), ncol=1)

plt.tight_layout()
save_fig(fanda, name="images/abl_gradnorm", format="svg")
close_fig(fanda)
