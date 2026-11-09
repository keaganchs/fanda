"""Shared run lists, metric keys, palette, loading and saving for the P0 ball-policy figures.

Every script does `from common import *`; this chdirs into pink/ so the fanda W&B cache (./.fanda.cache) and out/
land here whatever directory the script is started from.
"""
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent.parent))  # fanda is not installed into the venv (no build-system)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, ticker

import fanda  # noqa: F401  registers fanda's stylesheets (neurips, icml, iclr, slides, poster)
from fanda import transforms
from fanda.utils import close_fig, save_fig
from fanda.visualizations import add_lineplot, annotate_axis, decorate_axis, subplots
from fanda.wandb_client import fetch_wandb

ENTITY, PROJECT = "pink_robotics", "p0_manipulation"

# Resume chains, oldest first (from reports/rigab961_2026_11_05/pull_wandb.py). Each stage resumed from the previous
# stage's last checkpoint and W&B _step = training iteration continues across stages.
LINEAGE = {
    "189331": [
        "p0_manip_ballrigcont_r5balllong_s3_738",
        "p0_manip_ballrigcont_r6ft738_s0_742",
        "p0_manip_ballrigcont_r7_ft742_s0_758",
        "p0_manip_ballrigcont_r8_ft758_s0_764",
        "p0_manip_ballrigcont_r10_ft764_s0_800",
        "p0_manip_ballrigcont_r11_ft800_s0_824",
        "p0_manip_ballrigcont_r12_ft824_s1_847",
        "p0_manip_ballrigcont_r13wy_ft847_s1_852",
        "p0_manip_rewab_A_s0_935",
        "p0_manip_sacft_explore_189331",
    ],
    "961": ["p0_manip_ballrig_pincer_bc_aim0_s0_961"],
}
# Rig A/B fine-tunes: A = rig data + sim, B = sim only; 6 seeds each, 100 iterations from the base checkpoint.
AB = {
    "189331": [f"p0_manip_rigab_{a}_s{s}" for a in "AB" for s in range(6)],
    "961": [f"p0_manip_rigab961_{a}_s{s}" for a in "AB" for s in range(6)],
}
ARM = {"A": "A: rig data + sim", "B": "B: sim only"}
# The resumed tendon run is 3 W&B runs with one name (preempted twice); 981 is the same recipe without stu3emp.
TENDON = ["p0_manip_ballrig_tendon_bc_aim0_stu3emp_s0", "p0_manip_ballrig_tendon_bc_aim0_s0_981"]

# key -> (axis label, is a rate in [0, 1])
METRICS = {
    "Train/mean_reward": ("mean episode reward", False),
    "Curriculum/attempts/lift_rate": ("lift rate", True),
    "Curriculum/attempts/put_rate": ("put-back rate", True),
    "Curriculum/attempts/put_given_lift": ("put-back given lift", True),
    "Curriculum/press_trip_g/pressed_rate": ("press-trip rate", True),
    "Train/mean_episode_length": ("episode length (steps)", False),
    # not in the default grid
    "Metrics/ball/true_lifts": ("true lifts / episode", False),
    "Curriculum/attempts/press_p90_g": ("press p90 (g)", False),
    "Curriculum/attempts/t_lift_s": ("time to lift (s)", False),
}
GRID = list(METRICS)[:6]  # the 2x3 small-multiples order

PINK, DARK, GREY = "#E6007E", "#111111", "#6B6B6B"
PINK_TINT, GREY_TINT = "#F5A3CF", "#C4C4C4"
ARM_COLORS = {ARM["A"]: PINK, ARM["B"]: GREY}
ARM_COLORS_DARK = {ARM["A"]: PINK, ARM["B"]: "#A8A8A8"}  # #6B6B6B is too faint on #111111

for _f in Path.home().glob(".local/share/fonts/Montserrat-*.ttf"):
    font_manager.fontManager.addfont(str(_f))

_BRAND = {"axes.prop_cycle": plt.cycler(color=[PINK, DARK, GREY, PINK_TINT, GREY_TINT]), "axes.spines.top": False,
          "axes.spines.right": False, "legend.frameon": False}
_DARK = {"figure.facecolor": DARK, "axes.facecolor": DARK, "savefig.facecolor": DARK, "axes.edgecolor": "#BBBBBB",
         "axes.labelcolor": "#EEEEEE", "axes.titlecolor": "#EEEEEE", "xtick.color": "#BBBBBB",
         "ytick.color": "#BBBBBB", "text.color": "#EEEEEE", "grid.color": "#444444"}
# style name -> (fanda stylesheet, extra rc); "slides" uses Montserrat, the paper style uses LaTeX serif
STYLES = {
    "slides": ("slides", {**_BRAND, "font.family": "Montserrat"}),
    "dark": ("slides", {**_BRAND, **_DARK, "font.family": "Montserrat"}),
    # neurips sets text.usetex, but this laptop's TeX lacks cm-super (type1ec.sty): mathtext STIX serif instead
    "paper": ("neurips", {**_BRAND, "text.usetex": False, "font.serif": ["STIXGeneral", "DejaVu Serif"],
                          "mathtext.fontset": "stix"}),
}


def use_style(name):
    """Apply a fanda stylesheet plus the Pink palette. Returns the colour for 'neutral' lines in that style."""
    plt.rcdefaults()
    sheet, rc = STYLES[name]
    plt.style.use([sheet, rc])
    return "#EEEEEE" if name == "dark" else DARK


def load(names, key, samples=1000):
    """History of one metric for the named runs (fanda fetch_wandb, joblib-cached in ./.fanda.cache), with
    env_steps = (iteration - first iteration of the run set) x num_envs x steps_per_env and hours = _runtime/3600,
    both cumulative across the stages of a resume chain (stages ordered by `names`).
    One key per fetch: W&B's sampled history only returns rows where all requested keys are present."""
    df = fetch_wandb(ENTITY, PROJECT, keys=[key, "_runtime"], filters={"display_name": {"$in": list(names)}},
                     samples=samples)
    df = df.dropna(subset=[key]).copy()
    if key.startswith("Curriculum/attempts/"):  # attempt stats restart with every run: 0 until an attempt finishes
        df = df[df.groupby("run_id")[key].transform(lambda y: (y > 0).cummax())]
    df["per_iter"] = df["env.num_envs"] * df["ppo.num_steps_per_env"]
    df["stage"] = df["run_name"].map({n: i for i, n in enumerate(names)})
    return df.sort_values(["stage", "_step"]).reset_index(drop=True)


def chain(df, key):
    """Stitch a resume chain (rows of possibly several W&B runs) into one curve: later runs win where steps overlap,
    env steps and hours accumulate across runs. Adds columns env_steps and hours."""
    runs = df.groupby("run_id")["_step"].min().sort_values().index
    parts, steps0, hours0, it0 = [], 0.0, 0.0, None
    for i, rid in enumerate(runs):
        g = df[df.run_id == rid].sort_values("_step").copy()
        if i + 1 < len(runs):  # drop what the next run redid after resuming from an earlier checkpoint
            g = g[g._step < df[df.run_id == runs[i + 1]]._step.min()]
        if g.empty:
            continue
        it0 = g._step.iloc[0] if it0 is None else it0
        g["env_steps"] = steps0 + (g._step - it0) * g.per_iter
        g["hours"] = hours0 + g._runtime / 3600
        steps0, hours0, it0 = g.env_steps.iloc[-1], g.hours.iloc[-1], g._step.iloc[-1]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def per_run(df):
    """For independent runs (A/B seeds): env steps and hours since each run's own start."""
    df = df.copy()
    df["env_steps"] = (df._step - df.groupby("run_id")._step.transform("min")) * df.per_iter
    df["hours"] = df._runtime / 3600
    return df


def ab(exp, key, alpha=1.0):
    """A/B fine-tunes of base `exp` ("189331" / "961") for one metric: per-run env steps, EMA, an `arm` column, and
    `hours` averaged per (arm, iteration) over the seeds (fanda align_column) so wall-clock curves can be aggregated."""
    df = smooth(per_run(load(AB[exp], key)), key, alpha)
    df["arm"] = df.run_name.str.extract(r"_([AB])_s\d$")[0].map(ARM)
    df["seed"] = df.run_name.str[-1]
    df["hours_mean"] = transforms.align_column(df, "hours", groupby="arm")["hours"]
    return df


def smooth(df, key, alpha, groupby="run_id"):
    """fanda's EMA (alpha=1: raw). Returns a copy."""
    return transforms.exponential_moving_average(df.copy(), column=key, alpha=alpha, groupby=groupby)


def format_axis(ax, key, xlabel, ylabel=None):
    """Label + rate axes in %, via fanda's annotate/decorate."""
    f = fanda.fanda.Fanda(fig=ax.figure, ax=ax)
    f.pipe(annotate_axis, xlabel=xlabel, ylabel=ylabel or METRICS[key][0], labelsize="medium", grid_alpha=0.15)
    f.pipe(decorate_axis, wrect=4, hrect=4, ticklabelsize="small")
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    if METRICS.get(key, ("", False))[1]:
        ax.set_ylim(0, None if "press" in key else 1)  # press-trip share stays < ~25%: let it zoom
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    return ax


def millions(ax, scale=1e6, unit="M"):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / scale:g}{unit}"))


def save(f, name):
    """PDF + PNG into out/ via fanda.utils.save_fig, then close."""
    for fmt in ("pdf", "png"):
        f.pipe(save_fig, name=f"out/{name}", format=fmt)
    return f.pipe(close_fig)
