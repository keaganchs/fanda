"""Rig A/B fine-tunes, small multiples: 6 metrics over env steps, mean +- 95% bootstrap CI over the 6 seeds per arm.
One figure per base (189331, 961) and style (slides, dark, paper) -> out/ab_grid_<base>_<style>.{pdf,png}.
Plus out/ab_extras_<base>_slides: true lifts / episode, press p90, time to lift."""
from matplotlib.lines import Line2D

from common import *

ALPHA = 0.3  # EMA; the runs are only 100 iterations, keep it light


def grid(exp, style, keys, name):
    fg = use_style(style)
    paper = style == "paper"
    pal = ARM_COLORS_DARK if style == "dark" else ARM_COLORS
    rows = (len(keys) + 2) // 3
    f = subplots(rows, 3, figsize=(5.5, 1.7 * rows) if paper else (12, 3.2 * rows))
    for i, key in enumerate(keys):
        f.select(i).pipe(add_lineplot, df=ab(exp, key, ALPHA), x="env_steps", y=key, hue="arm", palette=pal,
                         errorbar=("ci", 95), err_kws={"alpha": 0.2, "lw": 0}, legend=False, lw=1.2 if paper else 2)
        format_axis(f.ax, key, "")
        millions(f.ax)
    f.fig.supxlabel("env steps since fine-tune start", fontsize="medium")
    handles = [Line2D([], [], color=c, lw=2, label=l) for l, c in pal.items()]
    f.axs.flat[min(3, len(keys) - 1)].legend(handles=handles, loc="lower right", fontsize="small")  # a flat panel
    f.fig.suptitle(f"Rig A/B fine-tunes of {exp}: mean and 95% CI over 6 seeds", x=0.01, ha="left",
                   fontsize="medium" if paper else "large", color=fg)
    save(f, name)


for exp in AB:
    for style in STYLES:
        grid(exp, style, GRID, f"ab_grid_{exp}_{style}")
    grid(exp, "slides", list(METRICS)[6:], f"ab_extras_{exp}_slides")
