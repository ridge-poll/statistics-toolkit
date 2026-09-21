"""
Power Analysis for Light-Activation-Normalized SD Metrics
===========================================================

Estimates statistical power for the paired reference-vs-treatment
comparisons produced by sd_analysis.py, based on either:
  - raw --events / --intervals CSVs (run through the same
    build_summary() pipeline used by sd_analysis.py), or
  - a previously saved --summary-csv (from `sd_analysis.py --save-csv`).

For each metric x treatment condition, this script:
  1. Runs a paired t-test on the pilot data itself and reports that
     p-value (this is "are we significant already", separate from the
     power projections below).
  2. Computes the paired effect size (Cohen's dz = mean(diff)/sd(diff))
     from the pilot data.
  3. Reports the achieved power at the pilot's current N (pairs).
  4. Solves for the N needed to reach one or more target power levels
     (default: 0.8), via a paired (one-sample-on-differences) t-test
     power model.
  5. Optionally plots a color-coded summary table (green/yellow/red by
     how many more recordings are needed) plus power-vs-N curves, one
     panel per metric.

This is meant to be run on an existing (possibly underpowered) pilot
dataset to decide how many more recordings/animals are needed for a
follow-up experiment -- it does not require re-running the raw
analysis unless you want to.

Caveats:
  - Cohen's dz from a small pilot is itself a noisy estimate; treat
    the required-N numbers as a rough planning aid, not a guarantee.
  - This uses a parametric (t-test) power model. If a metric is
    heavily non-normal or bounded (e.g. Trigger_Rate at 0%/100%),
    treat the numbers as approximate.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.power import TTestPower

from sd_functions import build_summary, get_condition_order


METRICS = [
    ("Trigger_Rate", "Trigger Rate (%)"),
    ("SDs_Per_Activation", "SDs per Activation"),
    ("AUC_Per_Event", "AUC per Event"),
    ("Time_To_First_SD", "Time to First SD (s)"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Power analysis for light-activation-normalized SD metrics."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary-csv", help="Previously saved sd_analysis.py summary CSV.")
    source.add_argument("--events", help="Path to SD-event CSV (recomputes summary from raw data).")

    parser.add_argument("--intervals", help="Path to light-activation interval CSV (required with --events).")
    parser.add_argument("--post-light-window", type=float, default=30,
                         help="Seconds after light turns off to still classify an SD as induced.")
    parser.add_argument("--sd-filter", choices=["induced", "spontaneous", "all"], default="induced",
                         help="Which SDs feed the metrics (only used with --events).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--power-targets", type=float, nargs="+", default=[0.8],
                         help="Target power level(s) to solve N for, e.g. --power-targets 0.8 0.9")
    parser.add_argument("--max-n", type=int, default=100,
                         help="Upper bound on N pairs to report/plot when solving for required sample size.")
    parser.add_argument("--no-plots", action="store_true", help="Skip the power-curve plots.")
    return parser.parse_args()


def load_summary(args):
    """Get the per-Recording/Condition summary either from a saved CSV
    or by recomputing it from raw events/intervals."""
    if args.summary_csv:
        return pd.read_csv(args.summary_csv)

    if not args.intervals:
        raise SystemExit("--intervals is required when using --events.")

    _, _, summary = build_summary(args.events, args.intervals, args.post_light_window, args.sd_filter)
    return summary


def paired_effect_size(control_vals, treat_vals):
    """Cohen's dz for a paired design: mean(diff) / sd(diff)."""
    diffs = np.asarray(treat_vals, dtype=float) - np.asarray(control_vals, dtype=float)
    n = len(diffs)
    if n < 2:
        return np.nan, n

    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return np.nan, n

    return float(np.mean(diffs) / sd), n


def paired_pvalue(control_vals, treat_vals):
    """Paired t-test p-value for the pilot data itself -- this is the
    significance the pilot data currently has, as opposed to the power
    projections below which are about a future/larger sample."""
    control_vals = np.asarray(control_vals, dtype=float)
    treat_vals = np.asarray(treat_vals, dtype=float)
    if len(control_vals) < 2:
        return np.nan

    _, p = stats.ttest_rel(control_vals, treat_vals, nan_policy="omit")
    return float(p) if not np.isnan(p) else np.nan


def solve_required_n(dz, alpha, power_target, max_n):
    """Smallest N (>=2, capped at max_n) reaching power_target, or None."""
    if dz is None or np.isnan(dz) or dz == 0:
        return None

    analysis = TTestPower()
    try:
        n = analysis.solve_power(effect_size=abs(dz), alpha=alpha, power=power_target, alternative="two-sided")
    except Exception:
        return None

    n_req = max(2, int(np.ceil(n)))
    return n_req if n_req <= max_n else None


def analyze_metric(summary, reference, treatments, value_column, alpha, power_targets, max_n):
    """One result dict per (treatment, metric) comparison."""
    analysis = TTestPower()
    control = summary[summary["Condition"] == reference][["Recording", value_column]].dropna()

    results = []
    for cond in treatments:
        treat = summary[summary["Condition"] == cond][["Recording", value_column]].dropna()
        merged = pd.merge(control, treat, on="Recording", suffixes=("_control", "_treat")).dropna()

        dz, n_pairs = paired_effect_size(
            merged[f"{value_column}_control"].values,
            merged[f"{value_column}_treat"].values,
        )

        p_value = paired_pvalue(
            merged[f"{value_column}_control"].values,
            merged[f"{value_column}_treat"].values,
        )

        if n_pairs >= 2 and not np.isnan(dz) and dz != 0:
            achieved_power = float(analysis.power(effect_size=abs(dz), nobs=n_pairs, alpha=alpha))
        else:
            achieved_power = np.nan

        required_n = {t: solve_required_n(dz, alpha, t, max_n) for t in power_targets}

        results.append({
            "Condition": cond,
            "N_Pairs": n_pairs,
            "P_Value": p_value,
            "Cohens_dz": dz,
            "Achieved_Power": achieved_power,
            "Required_N": required_n,
            "_max_n": max_n,
        })

    return results


def print_report(all_results, alpha, power_targets):
    print("\n" + "=" * 70)
    print("POWER ANALYSIS (paired t-test on differences, Cohen's dz)")
    print(f"alpha = {alpha}")
    print("=" * 70)

    for metric_label, metric_results in all_results.items():
        print(f"\n{metric_label.upper()}")
        print("-" * 70)

        header = ["Condition vs ref", "N pairs (pilot)", "p (current)", "Cohen's dz", "Achieved power"]
        header += [f"N needed (power={t})" for t in power_targets]
        print("\t".join(header))

        for r in metric_results:
            p_val = r["P_Value"]
            sig_flag = "  SIGNIFICANT" if (not np.isnan(p_val) and p_val < alpha) else ""

            row = [
                r["Condition"],
                str(r["N_Pairs"]),
                f"{p_val:.4f}" if not np.isnan(p_val) else "NA",
                f"{r['Cohens_dz']:.3f}" if not np.isnan(r["Cohens_dz"]) else "NA",
                f"{r['Achieved_Power']:.3f}" if not np.isnan(r["Achieved_Power"]) else "NA",
            ]
            for target in power_targets:
                n_req = r["Required_N"][target]
                row.append(str(n_req) if n_req is not None else f">{r['_max_n']}")
            print("\t".join(row) + sig_flag)

    print("\n" + "=" * 70)


def build_display_table(all_results, primary_target, max_n):
    """
    Flatten the per-metric results into one dataframe for the summary
    table plot -- one row per (metric, treatment) comparison.

    n (needed) / n (MORE needed) are reported against primary_target
    (by default the first value passed to --power-targets); if several
    target power levels were requested, the console report and power
    curves still cover all of them, but the table only has room for one.

    Cohen's dz is deliberately left out here (it's in the console
    report); Units are also left out since these metrics don't share a
    single unit system the way the original per-recording metrics do.
    """
    rows = []

    for metric_label, metric_results in all_results.items():
        for r in metric_results:
            n_have = r["N_Pairs"]
            n_needed = r["Required_N"][primary_target]

            if n_needed is None:
                n_needed_str = f">{max_n}"
                n_more_str = "NA"
                n_more_raw = np.nan
            else:
                n_more_raw = max(0, n_needed - n_have)
                n_needed_str = str(n_needed)
                n_more_str = str(n_more_raw)

            p_raw = r["P_Value"]
            power_raw = r["Achieved_Power"]

            rows.append({
                "Treatment": r["Condition"],
                "Metric": metric_label,
                "n (have)": n_have,
                "p (current)": f"{p_raw:.4f}" if not np.isnan(p_raw) else "NA",
                "Power (current)": f"{power_raw:.3f}" if not np.isnan(power_raw) else "NA",
                "n (needed)": n_needed_str,
                "n (MORE needed)": n_more_str,
                "_p_raw": p_raw,
                "_n_more_raw": n_more_raw,
            })

    return pd.DataFrame(rows)


def plot_results_table(display_df, title, alpha, primary_target):
    """
    Render the summary table as a color-coded matplotlib figure:
    green rows are already significant or already fully powered,
    yellow rows are close (<=5 more recordings needed), red rows need
    substantially more.
    """
    display_cols = [
        "Treatment", "Metric", "n (have)",
        "p (current)", "Power (current)", "n (needed)", "n (MORE needed)",
    ]
    tbl = display_df[display_cols].copy()

    fig_h = max(3, 0.35 * len(tbl) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    table = ax.table(cellText=tbl.values, colLabels=tbl.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(tbl.columns))))

    for j in range(len(tbl.columns)):
        table[0, j].set_facecolor("#2d3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i, (_, row) in enumerate(display_df.iterrows(), start=1):
        p_raw = row["_p_raw"]
        n_more_raw = row["_n_more_raw"]

        already_sig = (not np.isnan(p_raw)) and p_raw < alpha
        fully_powered = (not np.isnan(n_more_raw)) and n_more_raw == 0

        if already_sig or fully_powered:
            bg = "#d4edda"
        elif not np.isnan(n_more_raw) and n_more_raw <= 5:
            bg = "#fff3cd"
        else:
            bg = "#f8d7da"

        for j in range(len(tbl.columns)):
            table[i, j].set_facecolor(bg)

    fig.suptitle(
        f"{title}\n(alpha={alpha}, target power={primary_target})",
        fontsize=13, fontweight="bold", y=0.98
    )

    legend_elements = [
        Patch(facecolor="#d4edda", label="Already significant / no more needed"),
        Patch(facecolor="#fff3cd", label="Close \u2014 \u22645 more recordings"),
        Patch(facecolor="#f8d7da", label="More recordings required"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower center",
        bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8, frameon=False
    )

    plt.tight_layout()
    plt.show()


def plot_power_curves(all_results, alpha, max_n, power_targets):
    metrics = list(all_results.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
    axes = axes[0]

    analysis = TTestPower()
    n_range = np.arange(2, max_n + 1)

    for ax, metric_label in zip(axes, metrics):
        for r in all_results[metric_label]:
            dz = r["Cohens_dz"]
            if np.isnan(dz) or dz == 0:
                continue
            powers = analysis.power(effect_size=abs(dz), nobs=n_range, alpha=alpha)
            ax.plot(n_range, powers, label=f"{r['Condition']} (dz={dz:.2f}, pilot N={r['N_Pairs']})")

        for target in power_targets:
            ax.axhline(target, color="gray", linestyle="--", linewidth=1)

        ax.set_title(metric_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("N pairs")
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    summary = load_summary(args)
    reference, treatments, _ = get_condition_order(summary)

    all_results = {}
    for value_column, label in METRICS:
        if value_column not in summary.columns:
            continue
        all_results[label] = analyze_metric(
            summary, reference, treatments, value_column,
            args.alpha, args.power_targets, args.max_n
        )

    print_report(all_results, args.alpha, args.power_targets)

    if not args.no_plots:
        primary_target = args.power_targets[0]
        display_df = build_display_table(all_results, primary_target, args.max_n)
        plot_results_table(
            display_df,
            "Power Analysis \u2014 Light-Activation-Normalized SD Metrics",
            args.alpha, primary_target
        )
        # plot_power_curves(all_results, args.alpha, args.max_n, args.power_targets)


if __name__ == "__main__":
    main()