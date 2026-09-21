"""
sd_functions.py
================

All the non-CLI logic for the light-activation-normalized SD analysis:
loading data, assigning SDs to light activations, computing metrics,
paired significance testing, console reporting, CSV export, and
plotting. Imported by both sd_analysis.py and power_analysis.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon


# ============================================================
# Data loading / prep
# ============================================================

def load_data(events_path, intervals_path):
    """Load the SD-event CSV and the light-activation interval CSV."""
    events = pd.read_csv(events_path)
    intervals = pd.read_csv(intervals_path)

    intervals = intervals.copy()
    intervals["Activation_Start"] = pd.to_numeric(intervals["Activation_Start"], errors="coerce")
    intervals["Activation_End"] = pd.to_numeric(intervals["Activation_End"], errors="coerce")
    intervals["Activation_Duration"] = intervals["Activation_End"] - intervals["Activation_Start"]

    return events, intervals


def restrict_intervals_to_events(events, intervals):
    """
    Keep only interval rows whose Recording appears in the events CSV.

    This lets a single master intervals CSV (covering every treatment
    group) be reused as-is across separate per-treatment events CSVs,
    without ever pulling in another treatment's recordings.
    """
    relevant_recordings = set(events["Recording"].unique())
    restricted = intervals[intervals["Recording"].isin(relevant_recordings)].copy()
    restricted = restricted.reset_index(drop=True)
    restricted["Interval_ID"] = restricted.index

    missing = relevant_recordings - set(restricted["Recording"].unique())
    if missing:
        print(
            "WARNING: the following Recordings appear in the events CSV "
            f"but have no matching rows in the intervals CSV: {sorted(missing)}"
        )

    return restricted


def get_condition_order(df):
    """
    Determine reference/treatment conditions from a dataframe's
    Condition column (the first value encountered is treated as the
    reference/control). Works on either the raw events dataframe or an
    already-computed summary dataframe, since both carry Condition.
    """
    reference = df["Condition"].iloc[0]
    conditions = list(pd.unique(df["Condition"]))
    treatments = [c for c in conditions if c != reference]
    order = [reference] + treatments
    return reference, treatments, order


# ============================================================
# Assign each SD to its triggering light activation
# ============================================================

def assign_sd_to_activation(events, intervals, post_light_window):
    """
    For every non-NaN SD event, find the most recent light activation
    (same Recording + Condition) that started at or before the SD's
    Latency1, then check whether the SD falls within that activation's
    window extended by post_light_window. If so, tag the SD with that
    activation's Interval_ID and classify it as "induced"; otherwise
    "spontaneous".
    """
    sd_df = events[events["MaxAmp1"].notna()].copy()
    sd_df["Latency1"] = pd.to_numeric(sd_df["Latency1"], errors="coerce")

    sd_df["Assigned_Interval_ID"] = np.nan
    sd_df["SD_Type"] = "spontaneous"

    for (recording, condition), rec_events in sd_df.groupby(["Recording", "Condition"]):
        rec_intervals = intervals[
            (intervals["Recording"] == recording) & (intervals["Condition"] == condition)
        ].sort_values("Activation_Start")

        if rec_intervals.empty:
            continue

        starts = rec_intervals["Activation_Start"].values
        ends = rec_intervals["Activation_End"].values
        interval_ids = rec_intervals["Interval_ID"].values

        for idx, row in rec_events.iterrows():
            latency = row["Latency1"]
            if pd.isna(latency):
                continue

            pos = np.searchsorted(starts, latency, side="right") - 1
            if pos < 0:
                continue

            window_end = ends[pos] + post_light_window
            if starts[pos] <= latency <= window_end:
                sd_df.at[idx, "Assigned_Interval_ID"] = interval_ids[pos]
                sd_df.at[idx, "SD_Type"] = "induced"

    return sd_df


def apply_sd_filter(sd_df, sd_filter):
    """Restrict to induced / spontaneous / all SDs per --sd-filter."""
    if sd_filter == "all":
        return sd_df
    return sd_df[sd_df["SD_Type"] == sd_filter].copy()


# ============================================================
# Metric computation
# ============================================================

def compute_activation_summary(intervals):
    """Per-Recording/Condition: N_Activations, Total_Light_Time."""
    return intervals.groupby(["Recording", "Condition"]).agg(
        N_Activations=("Interval_ID", "count"),
        Total_Light_Time=("Activation_Duration", "sum")
    ).reset_index()


def compute_metrics(sd_df, filtered_sd_df, activation_summary):
    """
    Combine SD events with per-recording activation info.

    N_SDs_total : total number of real SDs before applying --sd-filter.
    N_SDs_filt  : number of SDs remaining after applying --sd-filter.
    N_Triggered : number of distinct light activations associated with
                  the filtered SDs.
    """
    total_counts = sd_df.groupby(["Recording", "Condition"]).agg(
        N_SDs_total=("MaxAmp1", "count")
    ).reset_index()

    filtered_counts = filtered_sd_df.groupby(["Recording", "Condition"]).agg(
        N_SDs_filt=("MaxAmp1", "count"),
        AUC_Sum=("AUC1", "sum")
    ).reset_index()

    triggered = filtered_sd_df.dropna(subset=["Assigned_Interval_ID"])
    n_triggered = triggered.groupby(["Recording", "Condition"])["Assigned_Interval_ID"] \
        .nunique().reset_index(name="N_Triggered")

    summary = (
        activation_summary
        .merge(total_counts, on=["Recording", "Condition"], how="left")
        .merge(filtered_counts, on=["Recording", "Condition"], how="left")
        .merge(n_triggered, on=["Recording", "Condition"], how="left")
    )

    summary["N_SDs_total"] = summary["N_SDs_total"].fillna(0)
    summary["N_SDs_filt"] = summary["N_SDs_filt"].fillna(0)
    summary["AUC_Sum"] = summary["AUC_Sum"].fillna(0)
    summary["N_Triggered"] = summary["N_Triggered"].fillna(0)

    summary["Trigger_Rate"] = np.where(
        summary["N_Activations"] > 0,
        summary["N_Triggered"] / summary["N_Activations"] * 100,
        np.nan
    )

    summary["SDs_Per_Activation"] = np.where(
        summary["N_Activations"] > 0,
        summary["N_SDs_filt"] / summary["N_Activations"],
        np.nan
    )

    summary["SDs_Per_Time"] = np.where(
        summary["Total_Light_Time"] > 0,
        summary["N_SDs_filt"] / summary["Total_Light_Time"],
        np.nan
    )

    summary["AUC_Per_Event"] = np.where(
        summary["N_SDs_filt"] > 0,
        summary["AUC_Sum"] / summary["N_SDs_filt"],
        np.nan
    )

    return summary


def compute_time_to_first_sd(sd_df, intervals):
    """
    For each Recording/Condition, compute time from activation start to
    the first (filtered) SD for every light activation.

    Time_To_First_SD      : mean latency across successful activations.
                             NaN if no activation produced an SD.
    Time_To_First_SD_List : latency for every activation in
                             chronological order (NaN for failures).
    """
    rows = []

    for (recording, condition), rec_intervals in intervals.groupby(["Recording", "Condition"]):
        rec_intervals = rec_intervals.sort_values("Activation_Start")

        latencies = []
        for _, interval in rec_intervals.iterrows():
            interval_id = interval["Interval_ID"]
            activation_start = interval["Activation_Start"]

            matching_sds = sd_df[sd_df["Assigned_Interval_ID"] == interval_id]

            if matching_sds.empty:
                latencies.append(None)
            else:
                first_sd = matching_sds["Latency1"].min()
                latencies.append(float(first_sd - activation_start))

        valid_latencies = [x for x in latencies if x is not None]
        mean_latency = np.mean(valid_latencies) if valid_latencies else np.nan

        rows.append({
            "Recording": recording,
            "Condition": condition,
            "Time_To_First_SD": mean_latency,
            "Time_To_First_SD_List": latencies
        })

    return pd.DataFrame(rows)


def build_summary(events_path, intervals_path, post_light_window, sd_filter):
    """
    Run the full pipeline (load -> restrict -> assign -> filter ->
    metrics) and return (events, intervals, summary). This is the one
    function both sd_analysis.py and power_analysis.py call to go from
    raw CSVs to the per-Recording/Condition summary dataframe.
    """
    events, intervals = load_data(events_path, intervals_path)
    intervals = restrict_intervals_to_events(events, intervals)

    sd_df = assign_sd_to_activation(events, intervals, post_light_window)
    filtered_sd_df = apply_sd_filter(sd_df, sd_filter)

    activation_summary = compute_activation_summary(intervals)
    summary = compute_metrics(sd_df, filtered_sd_df, activation_summary)

    latency_summary = compute_time_to_first_sd(filtered_sd_df, intervals)
    summary = summary.merge(latency_summary, on=["Recording", "Condition"], how="left")

    return events, intervals, summary


# ============================================================
# Paired stats (shared by report + plots)
# ============================================================

def paired_stats(control_vals, treat_vals, use_wilcoxon):
    """Return a formatted p-value string for a paired comparison."""
    if len(control_vals) <= 1:
        return "n<2"

    if use_wilcoxon:
        try:
            _, pval = wilcoxon(control_vals, treat_vals, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            return "all equal"
    else:
        _, pval = ttest_rel(control_vals, treat_vals, nan_policy="omit")

    if np.isnan(pval):
        return "all equal"
    if pval < 0.0001:
        return f"p={pval:.3e}"
    return f"p={pval:.4f}"


# ============================================================
# Printed summary
# ============================================================

PANEL_SPECS = [
    ("Trigger_Rate", "Trigger Rate (%)"),
    ("SDs_Per_Activation", "SDs per Activation"),
    ("SDs_Per_Time", "SDs per Accumulated Light Time"),
    ("AUC_Per_Event", "AUC per Event"),
]

BOOKKEEPING_COLS = [
    "Recording", "Condition", "N_Activations", "Total_Light_Time",
    "N_SDs_total", "N_SDs_filt", "N_Triggered",
]

SUMMARY_CSV_COLS = [
    "Recording", "Condition", "Trigger_Rate", "SDs_Per_Activation",
    "AUC_Per_Event", "Time_To_First_SD", "N_Activations",
    "Total_Light_Time", "N_SDs_total", "N_SDs_filt",
    "Time_To_First_SD_List",
]


def print_metric_summary(summary, reference, treatments, order, use_wilcoxon):
    print("\n" + "=" * 70)
    print("LIGHT-ACTIVATION-NORMALIZED SD SUMMARY")
    print("=" * 70)

    print("\nBOOKKEEPING")
    print("-" * 70)
    print(
        summary[BOOKKEEPING_COLS]
        .sort_values(["Recording", "Condition"])
        .to_csv(index=False, sep="\t")
        .strip()
    )

    for value_column, title in PANEL_SPECS:
        _print_metric_panel(summary, value_column, title, reference, treatments, order, use_wilcoxon)

    print("\n" + "=" * 70)


def _print_metric_panel(summary, value_column, title, reference, treatments, order, use_wilcoxon):
    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)

    wide = summary.pivot(index="Recording", columns="Condition", values=value_column)
    existing_conditions = [cond for cond in order if cond in wide.columns]
    wide = wide.reindex(columns=existing_conditions)
    wide.columns.name = None
    wide = wide.reset_index()

    print(wide.to_csv(sep="\t", index=False, na_rep="", float_format="%.6g").strip())

    print("\nCondition summary:")
    print("Condition\tN\tMean\tSEM")

    for cond in existing_conditions:
        values = wide[cond].dropna().values
        n = len(values)
        if n == 0:
            mean, sem = np.nan, np.nan
        else:
            mean = np.mean(values)
            sem = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else 0
        print(f"{cond}\t{n}\t{mean:.6g}\t{sem:.6g}")

    control = summary[summary["Condition"] == reference][["Recording", value_column]].dropna()

    for cond in treatments:
        treat = summary[summary["Condition"] == cond][["Recording", value_column]].dropna()
        merged = pd.merge(control, treat, on="Recording", suffixes=("_control", "_treat")).dropna()

        p_text = paired_stats(
            merged[f"{value_column}_control"].values,
            merged[f"{value_column}_treat"].values,
            use_wilcoxon
        )

        print(f"\nPaired comparison: {reference} vs {cond}")
        print(f"N pairs\t{len(merged)}")
        print(f"Test\t{'Wilcoxon' if use_wilcoxon else 'Paired t-test'}")
        print(f"Result\t{p_text}")


def save_summary_csv(summary):
    """Prompt for a save location (via a Tk file dialog) and write the
    analysis summary CSV there."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    output_path = filedialog.asksaveasfilename(
        title="Save analysis summary",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )

    root.destroy()

    if not output_path:
        print("\nCSV save cancelled.")
        return

    summary[SUMMARY_CSV_COLS].sort_values(["Recording", "Condition"]).to_csv(
        output_path, index=False, float_format="%.8g"
    )

    print(f"\nSaved summary CSV: {output_path}")


# ============================================================
# Plotting (bar + SEM + paired lines + scatter + stats)
# ============================================================

PLOT_PANEL_SPECS = [
    ("Trigger_Rate", "Trigger Rate", "% activations w/ SD"),
    ("SDs_Per_Activation", "SDs per Activation", "SDs / activation"),
    # ("SDs_Per_Time", "SDs per Accumulated Light Time", "SDs / sec"),
    ("Time_To_First_SD", "Time to First SD", "Time from activation start (s)"),
    ("AUC_Per_Event", "AUC per Event", "mV\u00b7sec / event"),
]


def plot_metrics(summary, reference, treatments, order, my_palette, recording_offsets, use_wilcoxon, sd_filter):
    fig, axes = plt.subplots(1, len(PLOT_PANEL_SPECS), figsize=(16, 4))
    fig.suptitle(
        f"Light-Activation-Normalized SD Metrics (sd_filter = {sd_filter})",
        fontsize=14, fontweight="bold"
    )

    for ax, (value_column, title, ylabel) in zip(axes, PLOT_PANEL_SPECS):
        _plot_panel(
            ax, summary, value_column, title, ylabel,
            reference, treatments, order, my_palette,
            recording_offsets, use_wilcoxon
        )

    plt.tight_layout()
    plt.show()


def _plot_panel(ax, summary, value_column, title, ylabel, reference, treatments, order,
                 my_palette, recording_offsets, use_wilcoxon):
    means, sems = [], []
    for cond in order:
        values = summary[summary["Condition"] == cond][value_column].dropna().values
        if len(values) > 0:
            means.append(np.mean(values))
            sems.append(np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0)
        else:
            means.append(np.nan)
            sems.append(np.nan)

    for i, cond in enumerate(order):
        ax.bar(i, means[i], width=0.4, color=my_palette[cond], alpha=0.25,
               edgecolor=my_palette[cond], linewidth=2)

    ax.errorbar(range(len(order)), means, yerr=sems, fmt="none", capsize=5, color="black")

    p_text = "n<2"
    control = summary[summary["Condition"] == reference][["Recording", value_column]].dropna()

    for cond in treatments:
        treat = summary[summary["Condition"] == cond][["Recording", value_column]].dropna()
        merged = pd.merge(control, treat, on="Recording", suffixes=("_control", "_treat")).dropna()

        x0 = order.index(reference)
        x1 = order.index(cond)

        for _, row in merged.iterrows():
            dx = recording_offsets.get(row["Recording"], 0)
            ax.plot(
                [x0 + dx, x1 + dx],
                [row[f"{value_column}_control"], row[f"{value_column}_treat"]],
                color="gray", linewidth=1, alpha=0.6, zorder=2
            )
            ax.scatter(x0 + dx, row[f"{value_column}_control"], color=my_palette[reference], s=40, zorder=3)
            ax.scatter(x1 + dx, row[f"{value_column}_treat"], color=my_palette[cond], s=40, zorder=3)

        p_text = paired_stats(
            merged[f"{value_column}_control"].values,
            merged[f"{value_column}_treat"].values,
            use_wilcoxon
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{p_text}", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
