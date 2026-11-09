# P0 ball-policy training curves (fanda, branch `pink`)

## Setup
```bash
cd ~/projects/pink/fanda          # branch pink
uv sync                           # .venv in the repo; leave the uv.lock change it makes uncommitted (git checkout uv.lock)
uv run python pink/<script>.py    # any cwd works: common.py chdirs into pink/
```
W&B login comes from the usual `wandb login` (~/.netrc). fanda isn't installed into the venv (pyproject has no
build-system), so `common.py` puts the repo root on `sys.path`.

## Scripts -> figures (pink/out/, PDF + PNG each)
| script | figures |
|---|---|
| `ab_grid.py` | `ab_grid_<base>_{slides,dark,paper}`: 2x3 small multiples, A vs B mean +- 95% bootstrap CI over 6 seeds; `ab_extras_<base>_slides`: true lifts/episode, press p90, time to lift |
| `ab_seeds.py` | `ab_seeds_{put,lift,reward}_ema{1,0.2}`: thin line per seed + bold arm mean, both bases side by side |
| `reward_time.py` | `reward_time_ab_<base>` (reward vs env steps and vs wall-clock minutes) and `reward_time_lineage_<base>` (full base training, vs env steps and hours) |
| `lineage.py` | `lineage_<base>_{steps,hours}_ema0.02`, `lineage_<base>_steps_ema0.2`: base policies' full resume chains, 2x3 grid, dotted = stage start |
| `tendon.py` | `tendon_{steps,hours}`: stu3emp s0 (3 W&B runs stitched) vs 981, 2x3 grid |

`<base>` is `189331` or `961`. Full render of everything: ~5 min with a warm cache.

## Runs (entity `pink_robotics`, project `p0_manipulation`) — lists in `common.py`
- `LINEAGE["189331"]`: 10 stages, `p0_manip_ballrigcont_r5balllong_s3_738` ... `p0_manip_rewab_A_s0_935`,
  `p0_manip_sacft_explore_189331` (taken from reports/rigab961_2026_11_05/pull_wandb.py, which walked `config.resume`).
  `LINEAGE["961"]`: one run from scratch, `p0_manip_ballrig_pincer_bc_aim0_s0_961`.
- `AB[base]`: `p0_manip_rigab_{A,B}_s{0..5}` (base 189331, iterations 12991-13090) and `p0_manip_rigab961_{A,B}_s{0..5}`
  (base 961 checkpoint 5000, iterations 5000-5099). A = rig data + sim, B = sim only. 100 iterations each.
- `TENDON`: `p0_manip_ballrig_tendon_bc_aim0_stu3emp_s0` = 3 W&B runs with that name (xd2hota6 0-619, 786dnw97 to 1662,
  u3kr4c2k to 5999; resumed from model_latest) + `p0_manip_ballrig_tendon_bc_aim0_s0_981`. A 4th early attempt named
  `..._stu3emp_s0_1092` (crashed at it 106) is not used.
- To find new runs: `wandb.Api().runs("pink_robotics/p0_manipulation", filters={"display_name": {"$regex": "..."}})`, or
  by `group`; to find a lineage, follow `run.config["resume"]` (runs/<run>/model_*.pt, or `<id>_<it>_*.pt` copies).

## Metric keys (`METRICS` in common.py)
- `Train/mean_reward`: mean episode reward (rsl-rl). `Train/mean_episode_length`: in policy steps (50 Hz).
- `Curriculum/attempts/lift_rate`, `put_rate`, `put_given_lift`: per batch of 100 finished lift attempts
  (p0_grasping/tasks/ball/attempts.py `summary`): lift success is scale-based (reading below on_g while held and raised),
  put rate is over all attempts, put_given_lift = put-backs / lifts (logged directly, no need to compute).
  These restart at 0 with every W&B run until the first batch closes: `load` drops the leading zeros per run.
- `Curriculum/press_trip_g/pressed_rate`: EMA share of ending episodes that ended by a press trip (scale.py).
- Extras: `Metrics/ball/true_lifts` (true-state lifts per episode, not scale-based), `Curriculum/attempts/press_p90_g`,
  `Curriculum/attempts/t_lift_s`. Other useful ones: `Metrics/ball/true_cycles`, `Metrics/ball/press_trips`,
  `Episode_Reward/*` (reward terms), `Gate/*` (SAC explore/gate per-attempt rates), `Perf/total_fps`.
- x axes: env steps = (iteration - first iteration) x `env.num_envs` x `ppo.num_steps_per_env` (98304 per iteration
  for 189331, 49152 for 961), cumulative across resume stages; wall-clock = W&B `_runtime` summed over stages (queue
  time and preemption gaps not counted). A/B wall-clock x = per-iteration mean over the arm's seeds (fanda
  `transforms.align_column`) so CIs can be computed.

## Caching / refresh
`fanda.wandb_client.fetch_wandb` is joblib-cached in `pink/.fanda.cache/` (git-ignored), keyed on all its arguments
(entity, project, keys, filters, samples). `common.load` fetches one key per call (W&B's sampled history only returns
rows where *all* requested keys exist), filtered by `display_name $in <run list>`, `samples=1000`.
- Cached results never expire: a still-running run stays frozen at the first fetch. Refresh with `rm -rf pink/.fanda.cache`
  (or delete the `fetch_wandb` subfolder), or change the run list / samples.
- The cache is ~800 MB: fetch_wandb merges the full flattened config (~1450 columns) into every history row.
  Fine for these runs; for many more runs, fetch fewer at a time.
- `samples=1000` is W&B's down-sampled history: runs longer than 1000 iterations are sub-sampled (good enough for curves).

## Adding a graphic
Copy the closest script (`ab_grid.py` for A/B CIs, `lineage.py`/`tendon.py` for single-run curves), then:
```python
from common import *                      # fanda + palette + loaders, chdir to pink/
use_style("slides")                      # or "dark" / "paper"
df = ab("961", "Curriculum/attempts/put_rate", alpha=0.3)       # A/B seeds, columns arm, env_steps, hours_mean
# or: chain(load(names, key), key)        # one resume chain -> env_steps, hours
f = subplots(1, 2, figsize=(12, 4.4))    # fanda Fanda: .select(i), .pipe(add_lineplot, ...)
f.select(0).pipe(add_lineplot, df=df, x="env_steps", y=key, hue="arm", palette=ARM_COLORS, errorbar=("ci", 95))
format_axis(f.ax, key, "env steps"); millions(f.ax)
save(f, "my_figure")                     # out/my_figure.pdf + .png
```
New metric: add it to `METRICS` (label, is-rate). Render the PNG and look at it.

## Style
- Palette: pink `#E6007E` (A arm / main line), dark `#111111`, grey `#6B6B6B` (B arm), tints `#F5A3CF`, `#C4C4C4`
  (raw/faint lines); on dark backgrounds B uses `#A8A8A8`.
- `slides`: fanda `slides.mplstyle` + Montserrat (from ~/.local/share/fonts). `dark`: same on #111111.
  `paper`: fanda `neurips.mplstyle` with usetex off and STIX serif (see pitfalls).
- Rates shown as % on 0-100% (press-trip rate auto-zooms); EMA via fanda `transforms.exponential_moving_average`
  (alpha=1 raw; 0.2-0.3 for 100-iteration A/B runs; 0.02 for long lineages).

## Pitfalls hit
- fanda's paper styles set `text.usetex: True`; this laptop's TeX lacks cm-super (`type1ec.sty`), so rendering fails.
  `STYLES["paper"]` turns usetex off. With usetex on, escape `%`/`_` in labels.
- fanda's visualizations module sets a serif font at import; `use_style` calls `plt.rcdefaults()` first.
- `fanda.utils.save_fig` paths are cwd-relative, as is the joblib cache: hence the chdir in common.py.
- Lineage stages each restart their statistics (episode lengths, attempt batches, press EMA): `lineage.py` restarts the
  EMA per stage and leaves out the first 30 iterations of later stages from the bold line (raw line keeps them).
- `fig.supxlabel` collides with a figure legend at `outside lower center`: put the legend in a flat panel instead.
- `uv sync` rewrites uv.lock; it's not part of these commits.
