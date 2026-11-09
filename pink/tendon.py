"""Tendon rig runs: p0_manip_ballrig_tendon_bc_aim0_stu3emp_s0 (preempted twice: 3 W&B runs with that name, stitched
by `chain`, later runs win on overlapping iterations) vs p0_manip_ballrig_tendon_bc_aim0_s0_981.
-> out/tendon_<x>.{pdf,png}, 6-metric grid over env steps and over wall-clock hours."""
from matplotlib.lines import Line2D

from common import *

COLORS = dict(zip(TENDON, [PINK, DARK]))
LABELS = dict(zip(TENDON, ["stu3emp s0 (resumed run)", "981"]))

for xname, (x, xlabel) in {"steps": ("env_steps", "env steps"),
                           "hours": ("hours", "wall-clock (h, sum of W&B runtimes)")}.items():
    use_style("slides")
    f = subplots(2, 3, figsize=(13, 6.4))
    for i, key in enumerate(GRID):
        f.select(i)
        df = load(TENDON, key)
        for name in TENDON:
            run = chain(df[df.run_name == name], key)
            f.ax.plot(run[x], run[key], color=COLORS[name], lw=0.4, alpha=0.3)
            run = smooth(run, key, 0.03, groupby="run_name")
            f.ax.plot(run[x], run[key], color=COLORS[name], lw=1.8)
        format_axis(f.ax, key, "")
        if xname == "steps":
            millions(f.ax)
    f.fig.supxlabel(xlabel, fontsize="medium")
    f.axs.flat[3].legend(handles=[Line2D([], [], color=COLORS[n], lw=2, label=LABELS[n]) for n in TENDON],
                         loc="lower right", fontsize="small")
    f.fig.suptitle("Tendon rig: stu3emp vs 981 (raw faint, EMA bold)", x=0.01, ha="left", fontsize="large")
    save(f, f"tendon_{xname}")
