"""
power_analysis.py

Standalone power analysis for experimental event recordings.
Loads the same CSV format used by paired_condition_analysis.py and
paired_delta_analysis.py, then reports — per metric and condition pair —
how many MORE recordings are needed to reach significance.

Usage:
    python power_analysis.py

Dependencies:
    pip install pandas numpy scipy matplotlib statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.power import TTestPower
import tkinter as tk
from tkinter import filedialog


# ── Config ───────────────────────────────────────────────────────────────────
ALPHA       = 0.05   # significance threshold
POWER       = 0.80   # desired power (1 - β)
MAX_SEARCH  = 200    # baseline upper bound for power-curve x-axis

# Start_s / End_s in the CSV are in seconds (column names end in _s).
# Set TIME_SCALE = 1 when timestamps are already in seconds (default).
# Change to 1000 if your pipeline ever switches to milliseconds.
TIME_SCALE  = 1      # divisor to convert timestamp units → seconds

METRICS     = ["Rate", "Amplitude", "Duration", "AUC"]
METRIC_UNITS = {
    "Rate":      "events/sec",
    "Amplitude": "mV",
    "Duration":  "sec",
    "AUC":       "mV·sec",
}

# Preferred name for the control condition.
# If this exact string is not found, reference is chosen as the condition
# with the most paired recordings (with a printed notice).
PREFERRED_REFERENCE = "ACSF"
# ─────────────────────────────────────────────────────────────────────────────

_solver = TTestPower()


def detect_reference(summary: pd.DataFrame) -> str:
    """
    Identify the control/reference condition.

    Priority:
      1. PREFERRED_REFERENCE if present in the data.
      2. The condition paired with the most recordings (most-data heuristic).
         A notice is printed so the choice is never silent.
    """
    conditions = pd.unique(summary["Condition"])

    if PREFERRED_REFERENCE in conditions:
        return PREFERRED_REFERENCE

    # Fallback: condition that appears in the most recordings
    counts = summary.groupby("Condition")["Recording"].nunique()
    reference = counts.idxmax()
    print(
        f"[NOTE] '{PREFERRED_REFERENCE}' not found in data. "
        f"Using '{reference}' as reference (most recordings: {counts[reference]})."
    )
    return reference


def load_and_summarize(file_path: str) -> pd.DataFrame:
    """Load CSV and compute per-recording per-condition summary metrics."""
    df = pd.read_csv(file_path)

    summary = df.groupby(["Recording", "Condition"]).agg(
        Amplitude  = ("MaxAmp1",   "mean"),
        Duration   = ("Duration1", "mean"),
        AUC        = ("AUC1",      "mean"),
        # Count only rows where MaxAmp1 is a valid (non-NaN) event value.
        # This correctly handles zero-event windows stored as NaN rows.
        EventCount = ("MaxAmp1",   lambda x: x.notna().sum()),
        Start      = ("Start_s",   "first"),
        End        = ("End_s",     "first"),
    ).reset_index()

    window_sec = (summary["End"] - summary["Start"]) / TIME_SCALE
    summary["Rate"] = np.where(
        (summary["EventCount"] > 0) & (window_sec > 0),
        summary["EventCount"] / window_sec,
        0,
    )

    return summary


def paired_differences(summary: pd.DataFrame, reference: str, treatment: str, metric: str) -> np.ndarray:
    """Return within-recording differences (treatment − control) for one metric."""
    ctrl = (summary[summary["Condition"] == reference]
            [["Recording", metric]]
            .rename(columns={metric: "ctrl"}))
    trt  = (summary[summary["Condition"] == treatment]
            [["Recording", metric]]
            .rename(columns={metric: "trt"}))
    merged = pd.merge(ctrl, trt, on="Recording").dropna()
    return (merged["trt"] - merged["ctrl"]).values


def required_n_paired(diffs: np.ndarray, alpha: float = ALPHA, power: float = POWER):
    """
    Compute the TOTAL n needed for a paired t-test at the observed effect size.

    Uses exact t-distribution via statsmodels TTestPower (not a normal
    approximation), which is accurate at the small sample sizes typical here.

    Returns an int, or np.nan when the calculation is not possible.
    """
    if len(diffs) < 2:
        return np.nan

    mu = np.mean(diffs)
    sd = np.std(diffs, ddof=1)

    if sd == 0:
        # Constant difference: if nonzero, any paired sample detects it.
        # Minimum valid paired t-test size is 2.
        return 2 if abs(mu) > 0 else np.nan

    d = abs(mu) / sd  # Cohen's dz for paired design

    # Suppress the convergence/precision warning statsmodels emits at
    # extreme effect sizes — these are expected and handled by the ceiling.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Iteration limit.*|.*xtol.*|.*converge.*",
            category=RuntimeWarning,
        )
        try:
            n_needed = _solver.solve_power(
                effect_size=d,
                alpha=alpha,
                power=power,
                alternative="two-sided",
            )
        except Exception:
            return np.nan

    return int(np.ceil(n_needed))


def power_at_n(diffs: np.ndarray, n: int, alpha: float = ALPHA) -> float:
    """
    Estimate achieved power of a paired t-test at a given n.

    Uses exact t-distribution via statsmodels TTestPower.
    """
    if len(diffs) < 2 or n < 2:
        return np.nan

    mu = np.mean(diffs)
    sd = np.std(diffs, ddof=1)

    if sd == 0:
        return 1.0 if abs(mu) > 0 else np.nan

    d = abs(mu) / sd

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Iteration limit.*|.*xtol.*|.*converge.*",
            category=RuntimeWarning,
        )
        try:
            return float(_solver.power(
                effect_size=d,
                nobs=n,
                alpha=alpha,
                alternative="two-sided",
            ))
        except Exception:
            return np.nan


def current_pvalue(diffs: np.ndarray) -> float:
    """One-sample t-test p-value against zero (equivalent to paired t-test)."""
    if len(diffs) < 2:
        return np.nan
    _, p = stats.ttest_1samp(diffs, 0, nan_policy="omit")
    return p


def build_results(summary: pd.DataFrame) -> pd.DataFrame:
    """Run power analysis for every treatment × metric combination."""
    reference  = detect_reference(summary)
    conditions = list(pd.unique(summary["Condition"]))
    treatments = [c for c in conditions if c != reference]

    rows = []
    for treatment in treatments:
        for metric in METRICS:
            diffs   = paired_differences(summary, reference, treatment, metric)
            n_have  = len(diffs)
            p_now   = current_pvalue(diffs)
            n_total = required_n_paired(diffs)
            pwr_now = power_at_n(diffs, n_have)

            if np.isnan(n_total) if isinstance(n_total, float) else False:
                n_more = np.nan
            elif isinstance(n_total, float) and np.isnan(n_total):
                n_more = np.nan
            else:
                n_more = max(0, n_total - n_have)

            rows.append({
                "Treatment":       treatment,
                "vs Control":      reference,
                "Metric":          metric,
                "Units":           METRIC_UNITS[metric],
                "n (have)":        n_have,
                "p (current)":     round(p_now,  4) if not np.isnan(p_now)  else np.nan,
                "Power (current)": round(pwr_now, 3) if not np.isnan(pwr_now) else np.nan,
                "n (needed)":      int(n_total) if not (isinstance(n_total, float) and np.isnan(n_total)) else np.nan,
                "n (MORE needed)": int(n_more)  if not (isinstance(n_more,  float) and np.isnan(n_more))  else np.nan,
                "_diffs":          diffs,
            })

    return pd.DataFrame(rows)


# ── Power curve plot ──────────────────────────────────────────────────────────

def plot_power_curves(results: pd.DataFrame, alpha: float = ALPHA, power_target: float = POWER):
    """One figure per treatment with one subplot per metric showing power vs n."""
    treatments = results["Treatment"].unique()
    reference  = results["vs Control"].iloc[0]

    base_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    pal = {t: c for t, c in zip(treatments, base_colors)}

    for treatment in treatments:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4), sharey=True)
        fig.suptitle(
            f"Power Analysis — {reference} vs {treatment}  "
            f"(α={alpha}, target power={power_target})",
            fontsize=13, fontweight="bold"
        )

        for ax, metric in zip(axes, METRICS):
            row      = results[(results["Treatment"] == treatment) & (results["Metric"] == metric)].iloc[0]
            diffs    = row["_diffs"]
            n_have   = row["n (have)"]
            n_needed = row["n (needed)"]
            color    = pal[treatment]

            # Expand x-range so the curve always extends past the needed n
            try:
                xmax = max(MAX_SEARCH, int(n_needed * 1.2) + 5)
            except (TypeError, ValueError):
                xmax = MAX_SEARCH

            ns     = np.arange(2, xmax + 1)
            powers = [power_at_n(diffs, int(n), alpha) for n in ns]

            ax.plot(ns, powers, color=color, lw=2)
            ax.axhline(power_target, ls="--", color="black", lw=1, label=f"Target {power_target}")
            ax.axvline(n_have, ls=":", color="gray", lw=1.5, label=f"Have n={n_have}")

            n_needed_valid = not (isinstance(n_needed, float) and np.isnan(n_needed))
            if n_needed_valid:
                ax.axvline(n_needed, ls="-", color=color, lw=1.5, alpha=0.7,
                           label=f"Need n={n_needed}")
                ax.scatter([n_needed], [power_target], color=color, zorder=5, s=60)

            # Shade the gap between current and needed n
            if n_needed_valid and n_needed > n_have:
                ax.axvspan(n_have, n_needed, alpha=0.08, color=color)

            # Annotate current power
            pwr_now = row["Power (current)"]
            pwr_valid = not (isinstance(pwr_now, float) and np.isnan(pwr_now))
            if pwr_valid:
                ax.scatter([n_have], [pwr_now], color="gray", zorder=5, s=60)
                ax.annotate(
                    f"  {pwr_now:.0%}",
                    (n_have, pwr_now),
                    fontsize=8, color="gray"
                )

            ax.set_title(f"{metric}\n({METRIC_UNITS[metric]})", fontsize=10, fontweight="bold")
            ax.set_xlabel("n (recordings)")
            if ax == axes[0]:
                ax.set_ylabel("Estimated Power")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, loc="lower right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()

    plt.show()


# ── Summary table plot ────────────────────────────────────────────────────────

def plot_summary_table(display_df: pd.DataFrame):
    """Render the results as a colour-coded matplotlib table."""
    cols = ["Treatment", "Metric", "Units", "n (have)", "p (current)",
            "Power (current)", "n (needed)", "n (MORE needed)"]
    tbl = display_df[cols].copy()

    fig_h = max(3, 0.15 * len(tbl) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText  = tbl.values,
        colLabels = tbl.columns,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(tbl.columns))))

    # Colour header
    for j in range(len(tbl.columns)):
        table[0, j].set_facecolor("#2d3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour rows by significance + adequacy
    for i, (_, row) in enumerate(tbl.iterrows(), start=1):
        p   = row["p (current)"]
        n_m = row["n (MORE needed)"]

        try:
            already_sig = float(p) < ALPHA
        except (ValueError, TypeError):
            already_sig = False

        try:
            more = int(float(n_m))
        except (ValueError, TypeError):
            more = 999

        if already_sig or more == 0:
            bg = "#d4edda"   # green — already significant / powered
        elif more <= 2:
            bg = "#fff3cd"   # yellow — close
        else:
            bg = "#f8d7da"   # red — more work needed

        for j in range(len(tbl.columns)):
            table[i, j].set_facecolor(bg)

    fig.suptitle(
        f"Power Analysis Summary  (α={ALPHA}, target power={POWER})",
        fontsize=13, fontweight="bold", y=0.98
    )

    legend_elements = [
        Patch(facecolor="#d4edda", label="Already significant / no more needed"),
        Patch(facecolor="#fff3cd", label="Close — ≤2 more recordings"),
        Patch(facecolor="#f8d7da", label="More recordings required"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"\nLoading: {file_path}")
    summary = load_and_summarize(file_path)

    reference  = detect_reference(summary)
    conditions = list(pd.unique(summary["Condition"]))
    treatments = [c for c in conditions if c != reference]

    print(f"Reference condition : {reference}")
    print(f"Treatment(s)        : {', '.join(treatments)}")
    print(f"Recordings found    : {summary['Recording'].nunique()}\n")

    results    = build_results(summary)
    display_df = results.drop(columns=["_diffs"])

    # ── Console summary ──────────────────────────────────────────────────────
    print("=" * 72)
    print(f"  POWER ANALYSIS  |  α={ALPHA}  |  target power={POWER}")
    print("=" * 72)
    for _, row in display_df.iterrows():
        sig_flag = ""
        try:
            if float(row["p (current)"]) < ALPHA:
                sig_flag = "  ✓ SIGNIFICANT"
        except (ValueError, TypeError):
            pass

        more = row["n (MORE needed)"]
        if isinstance(more, float) and np.isnan(more):
            more_str = "  (insufficient data)"
        elif int(more) == 0:
            more_str = "  → already powered / significant"
        else:
            more_str = f"  → need {int(more)} MORE recording(s)"

        print(
            f"  {row['Treatment']:12s} | {row['Metric']:10s}"
            f"  n={row['n (have)']:>2}  p={str(row['p (current)']):>7}"
            f"  power={str(row['Power (current)']):>5}"
            f"{more_str}{sig_flag}"
        )
    print("=" * 72)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_summary_table(display_df)
    plot_power_curves(results, alpha=ALPHA, power_target=POWER)


if __name__ == "__main__":
    main()