"""Rig A/B fine-tunes, per seed: thin line per seed + bold mean per arm, both bases side by side, one metric per figure.
-> out/ab_seeds_<metric>_ema<alpha>.{pdf,png} for put-back rate, lift rate and reward, at two smoothing levels."""
from matplotlib.lines import Line2D

from common import *

KEYS = {"put": "Curriculum/attempts/put_rate", "lift": "Curriculum/attempts/lift_rate", "reward": "Train/mean_reward"}


def fig(short, alpha):
    key = KEYS[short]
    use_style("slides")
    f = subplots(1, 2, figsize=(12, 4.6))
    for i, exp in enumerate(AB):
        df = ab(exp, key, alpha)
        f.select(i)
        f.pipe(add_lineplot, df=df, x="env_steps", y=key, hue="arm", units="run_name", estimator=None,
               palette=ARM_COLORS, lw=0.8, alpha=0.45, legend=False)
        f.pipe(add_lineplot, df=df, x="env_steps", y=key, hue="arm", palette=ARM_COLORS, errorbar=None, lw=2.6,
               legend=False)
        format_axis(f.ax, key, "env steps since fine-tune start")
        f.ax.set_title(f"fine-tunes of {exp}", loc="left", fontsize="medium")
        millions(f.ax)
    handles = [Line2D([], [], color=c, lw=2.6, label=l) for l, c in ARM_COLORS.items()]
    handles.append(Line2D([], [], color=GREY, lw=0.8, alpha=0.6, label="single seed"))
    f.axs.flat[1].legend(handles=handles, loc="lower right", fontsize="small")
    save(f, f"ab_seeds_{short}_ema{alpha:g}")


for short in KEYS:
    for alpha in (1.0, 0.2):
        fig(short, alpha)
