# SD / Light-Activation Pipeline

Three files, same CLI as the original script, plus a new power-analysis
tool. Keep the files together in one folder — they import each other
by filename, no packaging needed.

## Layout

| File               | Contents |
|---------------------|----------|
| `sd_functions.py`    | Everything that isn't CLI glue: loading/restricting CSVs, assigning SDs to light activations, computing metrics, paired stats, console reporting, CSV export, plotting |
| `sd_analysis.py`     | **Entry point for the main analysis.** Argument parsing + orchestration only — same flags, same behavior as before |
| `power_analysis.py`  | **New.** Entry point for sample-size / power estimation, built on the same `sd_functions.py` |

`sd_analysis.py` and `power_analysis.py` both import from
`sd_functions.py`, so any fix to the assignment/metric logic only
needs to happen in one place.

## Running the main analysis

Unchanged from before:

```bash
python sd_analysis.py --events events.csv --intervals intervals.csv
```

Flags: `--save-csv`, `--wilcoxon`, `--post-light-window`,
`--sd-filter {induced,spontaneous,all}`, `--no-plots`.

## Running the power analysis

Requires `statsmodels` (`pip install statsmodels --break-system-packages`
if you don't have it).

Feed it pilot data one of two ways:

```bash
# A. straight from raw CSVs (recomputes the summary via sd_functions.py)
python power_analysis.py --events events.csv --intervals intervals.csv

# B. from a summary CSV you already saved via sd_analysis.py --save-csv
python power_analysis.py --summary-csv my_summary.csv
```

Useful flags:

- `--power-targets 0.8 0.9` — solve required N for multiple power levels at once
- `--alpha 0.05` — significance level used for the power model
- `--max-n 100` — cap on how far the search/plot goes (reports `>max_n` if not reached)
- `--no-plots` — table only, no power-curve figure

### What it reports

For each metric (Trigger Rate, SDs per Activation, AUC per Event, Time
to First SD) and each treatment vs. the reference condition:

- **Cohen's dz** — the paired effect size from the pilot data
  (`mean(diff) / sd(diff)`)
- **Achieved power** at the pilot's current N
- **N needed** to reach each target power, via a paired t-test power
  model (`statsmodels.stats.power.TTestPower`)

A power-curve plot (power vs. N, one panel per metric) is shown unless
`--no-plots` is passed.

### Caveats worth keeping in mind

- Cohen's dz estimated from a small pilot (e.g. n=5) is itself noisy —
  treat the required-N numbers as a planning aid, not a guarantee.
- This is a parametric (t-test-based) power model. For metrics that
  are heavily non-normal or bounded (e.g. Trigger Rate stuck near
  0%/100%), the numbers are approximate; treat it as a starting point
  rather than a final justification in a grant/IACUC application.
