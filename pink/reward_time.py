"""Mean episode reward over env steps (left) and over wall-clock hours (right).
-> out/reward_time_ab_<base>.{pdf,png}: the A/B fine-tunes, mean +- 95% CI (x for wall-clock = per-iteration mean over seeds).
-> out/reward_time_lineage_<base>.{pdf,png}: the base policy's full resume chain (raw faint + EMA bold).
Wall-clock is W&B _runtime summed over the stages: queue time and preemption gaps are not counted."""
from matplotlib.lines import Line2D

from common import *

KEY = "Train/mean_reward"
use_style("slides")

for exp in AB:
    df = ab(exp, KEY, 0.3)
    df["minutes"] = df.hours_mean * 60  # a fine-tune is ~10-40 min
    f = subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for i, (x, label) in enumerate((("env_steps", "env steps since fine-tune start"),
                                    ("minutes", "wall-clock since fine-tune start (min)"))):
        f.select(i).pipe(add_lineplot, df=df, x=x, y=KEY, hue="arm", palette=ARM_COLORS, errorbar=("ci", 95),
                         err_kws={"alpha": 0.2, "lw": 0}, legend=False, lw=2)
        format_axis(f.ax, KEY, label, ylabel=None if i == 0 else "")
    millions(f.axs.flat[0])
    f.axs.flat[1].legend(handles=[Line2D([], [], color=c, lw=2, label=l) for l, c in ARM_COLORS.items()],
                         loc="lower right", fontsize="small")
    f.fig.suptitle(f"Rig A/B fine-tunes of {exp}: reward, mean and 95% CI over 6 seeds", x=0.01, ha="left")
    save(f, f"reward_time_ab_{exp}")

for exp, names in LINEAGE.items():
    raw = chain(load(names, KEY), KEY)
    sm = smooth(raw, KEY, 0.02, groupby="stage")
    f = subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for i, (x, label) in enumerate((("env_steps", "env steps"), ("hours", "wall-clock (h, W&B runtime)"))):
        f.select(i)
        f.ax.plot(raw[x], raw[KEY], color=GREY_TINT, lw=0.5)
        f.ax.plot(sm[x], sm[KEY], color=PINK, lw=2)
        format_axis(f.ax, KEY, label, ylabel=None if i == 0 else "")
    millions(f.axs.flat[0], 1e9, "B")
    f.fig.suptitle(f"Base policy {exp}: reward over its full training ({len(names)} stage{'s' * (len(names) > 1)})",
                   x=0.01, ha="left")
    save(f, f"reward_time_lineage_{exp}")
