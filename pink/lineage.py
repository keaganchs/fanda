"""Base policies' full training (the resume chain from scratch), small multiples of the 6 grid metrics.
Raw curve faint, EMA bold (restarted per stage, without the first WARMUP iterations of each later stage, since every
stage restarts its statistics); dotted verticals = stage starts.
-> out/lineage_<base>_<x>_ema<alpha>.{pdf,png} for x in env steps / wall-clock hours and two smoothing levels."""
from common import *

WARMUP = 30  # iterations of each later stage left out of the bold line (the raw line keeps them)
XS = {"steps": ("env_steps", "env steps"), "hours": ("hours", "wall-clock (h, sum of W&B runtimes)")}


def fig(exp, xname, alpha):
    x, xlabel = XS[xname]
    use_style("slides")
    f = subplots(2, 3, figsize=(13, 6.4))
    for i, key in enumerate(GRID):
        raw = chain(load(LINEAGE[exp], key), key)
        sm = smooth(raw, key, alpha, groupby="stage")
        f.select(i)
        for s in raw.groupby("stage")[x].min().iloc[1:]:
            f.ax.axvline(s, color=GREY_TINT, lw=0.6, ls=":", zorder=0)
        f.ax.plot(raw[x], raw[key], color=GREY_TINT, lw=0.4)
        for st, g in sm.groupby("stage"):  # no line jumping across stage starts
            if st:  # stats (episode lengths, attempt batches, press EMA) restart with each stage: skip its warm-up
                g = g[g._step >= g._step.min() + WARMUP]
            f.ax.plot(g[x], g[key], color=PINK, lw=1.8)
        format_axis(f.ax, key, "")
        if xname == "steps":
            millions(f.ax, 1e9, "B")
    f.fig.supxlabel(xlabel, fontsize="medium")
    n = len(LINEAGE[exp])
    stages = f"{n} resume stages, dotted = stage start" if n > 1 else "one run from scratch"
    f.fig.suptitle(f"Base policy {exp}: full training ({stages})", x=0.01, ha="left", fontsize="large")
    save(f, f"lineage_{exp}_{xname}_ema{alpha:g}")


for exp in LINEAGE:
    for xname in XS:
        fig(exp, xname, 0.02)
    fig(exp, "steps", 0.2)
