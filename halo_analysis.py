"""
Light-Activation-Normalized SD Analysis
========================================

Analyzes spreading depolarization (SD) events relative to Halorhodopsin
light stimulation, using the light activation itself (rather than raw
recording duration) as the normalization unit.

Inputs:
    --events     Path to the SD-event CSV. One row per SD (or a bookkeeping
                  row with MaxAmp1 = NaN indicating "no SD"). Must contain:
                  Recording, Condition, Latency1, MaxAmp1, AUC1
    --intervals  Path to the master light-activation CSV. One row per light
                  pulse. Must contain:
                  Recording, Condition, Activation_Start, Activation_End

Note: `Recording` values are assumed to already be globally unique across
treatment groups (e.g. "h40R3", "manR3"), so no separate pairing key is
needed -- merges are done directly on Recording.

Metrics computed per Recording (using only SDs selected by --sd-filter):
    1. Trigger rate (%)          - fraction of light activations that
                                     produced >=1 qualifying SD
    2. SDs per activation        - qualifying SD count / N activations
    3. SDs per accumulated time  - qualifying SD count / total light-on sec
    4. AUC per event             - mean AUC1 of qualifying SDs (0 if none)

Classification (induced vs. spontaneous):
    An SD is "induced" if its Latency1 falls within the most recent light
    activation's [Activation_Start, Activation_End + post_light_window]
    window for that Recording. Otherwise it is "spontaneous". The
    interval CSV is the sole source of truth for activation timing --
    Light_Status/Change_Time columns in the events CSV (if present) are
    not used here.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Light-activation-normalized SD analysis."
    )
    parser.add_argument(
        "--events",
        required=True,
        help="Path to SD-event CSV."
    )
    parser.add_argument(
        "--intervals",
        required=True,
        help="Path to light-activation interval CSV."
    )
    parser.add_argument(
        "--wilcoxon",
        action="store_true",
        help="Use Wilcoxon signed-rank test instead of paired t-test."
    )
    parser.add_argument(
        "--post-light-window",
        type=float,
        default=30,
        help="Seconds after light turns off to still classify an SD as induced."
    )
    parser.add_argument(
        "--sd-filter",
        choices=["induced", "spontaneous", "all"],
        default="induced",
        help="Which SDs feed the metrics: induced, spontaneous, or all."
    )
    return parser.parse_args()


# ============================================================
# Data loading / prep
# ============================================================

def load_data(events_path, intervals_path):
    events = pd.read_csv(events_path)
    intervals = pd.read_csv(intervals_path)

    intervals = intervals.copy()
    intervals["Activation_Start"] = pd.to_numeric(
        intervals["Activation_Start"], errors="coerce"
    )
    intervals["Activation_End"] = pd.to_numeric(
        intervals["Activation_End"], errors="coerce"
    )
    intervals["Activation_Duration"] = (
        intervals["Activation_End"] - intervals["Activation_Start"]
    )

    return events, intervals


def restrict_intervals_to_events(events, intervals):
    """
    The events CSV drives the analysis. Keep only interval rows whose
    Recording appears in the events CSV -- this lets a single master
    intervals CSV (covering every treatment group) be reused as-is
    across separate per-treatment events CSVs, without ever pulling in
    another treatment's recordings.
    """
    relevant_recordings = set(events["Recording"].unique())
    restricted = intervals[
        intervals["Recording"].isin(relevant_recordings)
    ].copy()
    restricted = restricted.reset_index(drop=True)
    restricted["Interval_ID"] = restricted.index

    missing = relevant_recordings - set(restricted["Recording"].unique())
    if missing:
        print(
            "WARNING: the following Recordings appear in the events CSV "
            f"but have no matching rows in the intervals CSV: {sorted(missing)}"
        )

    return restricted


def get_condition_order(events):
    """
    Reference/treatment conditions are determined from the events CSV
    (the driving input), not the intervals CSV, so a shared master
    intervals file never leaks conditions that aren't actually present
    in the events file being analyzed.
    """
    reference = events["Condition"].iloc[0]
    conditions = list(pd.unique(events["Condition"]))
    treatments = [c for c in conditions if c != reference]
    order = [reference] + treatments
    return reference, treatments, order


# ============================================================
# Assign each SD to its triggering light activation
# ============================================================

def assign_sd_to_activation(events, intervals, post_light_window):
    """
    For every non-NaN SD event, find the most recent light activation
    (same Recording) that started at or before the SD's Latency1, then
    check whether the SD falls within that activation's window extended
    by post_light_window. If so, tag the SD with that activation's
    Interval_ID and classify it as "induced"; otherwise "spontaneous".
    """
    sd_df = events[events["MaxAmp1"].notna()].copy()
    sd_df["Latency1"] = pd.to_numeric(sd_df["Latency1"], errors="coerce")

    sd_df["Assigned_Interval_ID"] = np.nan
    sd_df["SD_Type"] = "spontaneous"

    for recording, rec_events in sd_df.groupby("Recording"):
        rec_intervals = intervals[
            intervals["Recording"] == recording
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

            # Index of the last activation that started at or before latency
            pos = np.searchsorted(starts, latency, side="right") - 1
            if pos < 0:
                continue  # SD occurred before any activation started

            window_end = ends[pos] + post_light_window
            if starts[pos] <= latency <= window_end:
                sd_df.at[idx, "Assigned_Interval_ID"] = interval_ids[pos]
                sd_df.at[idx, "SD_Type"] = "induced"

    return sd_df


def apply_sd_filter(sd_df, sd_filter):
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


def compute_metrics(filtered_sd_df, activation_summary):
    """
    Combine filtered SD events with per-recording activation info to
    produce the four metrics of interest, one row per Recording/Condition.
    """
    sd_counts = filtered_sd_df.groupby(
        ["Recording", "Condition"]
    ).agg(
        N_SDs=("MaxAmp1", "count"),
        AUC_Sum=("AUC1", "sum")
    ).reset_index()

    triggered = filtered_sd_df.dropna(subset=["Assigned_Interval_ID"])
    n_triggered = triggered.groupby(
        ["Recording", "Condition"]
    )["Assigned_Interval_ID"].nunique().reset_index(
        name="N_Triggered"
    )

    summary = activation_summary.merge(
        sd_counts, on=["Recording", "Condition"], how="left"
    ).merge(
        n_triggered, on=["Recording", "Condition"], how="left"
    )

    summary["N_SDs"] = summary["N_SDs"].fillna(0)
    summary["AUC_Sum"] = summary["AUC_Sum"].fillna(0)
    summary["N_Triggered"] = summary["N_Triggered"].fillna(0)

    summary["Trigger_Rate"] = np.where(
        summary["N_Activations"] > 0,
        summary["N_Triggered"] / summary["N_Activations"] * 100,
        np.nan
    )

    summary["SDs_Per_Activation"] = np.where(
        summary["N_Activations"] > 0,
        summary["N_SDs"] / summary["N_Activations"],
        np.nan
    )

    summary["SDs_Per_Time"] = np.where(
        summary["Total_Light_Time"] > 0,
        summary["N_SDs"] / summary["Total_Light_Time"],
        np.nan
    )

    summary["AUC_Per_Event"] = np.where(
        summary["N_SDs"] > 0,
        summary["AUC_Sum"] / summary["N_SDs"],
        0
    )

    return summary


# ============================================================
# Plotting (bar + SEM + paired lines + scatter + stats)
# ============================================================

def paired_stats(control_vals, treat_vals, use_wilcoxon):
    if len(control_vals) <= 1:
        return "n<2"

    if use_wilcoxon:
        try:
            _, pval = wilcoxon(
                control_vals, treat_vals,
                zero_method="wilcox", alternative="two-sided"
            )
        except ValueError:
            return "all equal"
    else:
        _, pval = ttest_rel(control_vals, treat_vals, nan_policy="omit")

    if np.isnan(pval):
        return "all equal"
    return f"p={pval:.6f}"


def plot_metrics(summary, reference, treatments, order, my_palette,
                  recording_offsets, use_wilcoxon, sd_filter):
    panel_specs = [
        ("Trigger_Rate", "Trigger Rate", "% activations w/ SD"),
        ("SDs_Per_Activation", "SDs per Activation", "SDs / activation"),
        ("SDs_Per_Time", "SDs per Accumulated Light Time", "SDs / sec"),
        ("AUC_Per_Event", "AUC per Event", "mV\u00b7sec / event"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Light-Activation-Normalized SD Metrics (sd_filter = {sd_filter})",
        fontsize=14, fontweight="bold"
    )

    for ax, (value_column, title, ylabel) in zip(axes, panel_specs):
        means, sems = [], []

        for cond in order:
            values = summary[
                summary["Condition"] == cond
            ][value_column].dropna().values

            if len(values) > 0:
                means.append(np.mean(values))
                sems.append(
                    np.std(values, ddof=1) / np.sqrt(len(values))
                    if len(values) > 1 else 0
                )
            else:
                means.append(np.nan)
                sems.append(np.nan)

        for i, cond in enumerate(order):
            ax.bar(
                i, means[i], width=0.4,
                color=my_palette[cond], alpha=0.25,
                edgecolor=my_palette[cond], linewidth=2
            )

        ax.errorbar(
            range(len(order)), means, yerr=sems,
            fmt="none", capsize=5, color="black"
        )

        p_text = "n<2"
        control = summary[
            summary["Condition"] == reference
        ][["Recording", value_column]].dropna()

        for cond in treatments:
            treat = summary[
                summary["Condition"] == cond
            ][["Recording", value_column]].dropna()

            merged = pd.merge(
                control, treat, on="Recording",
                suffixes=("_control", "_treat")
            ).dropna()

            x0 = order.index(reference)
            x1 = order.index(cond)

            for _, row in merged.iterrows():
                dx = recording_offsets.get(row["Recording"], 0)

                ax.plot(
                    [x0 + dx, x1 + dx],
                    [row[f"{value_column}_control"], row[f"{value_column}_treat"]],
                    color="gray", linewidth=1, alpha=0.6, zorder=2
                )
                ax.scatter(
                    x0 + dx, row[f"{value_column}_control"],
                    color=my_palette[reference], s=40, zorder=3
                )
                ax.scatter(
                    x1 + dx, row[f"{value_column}_treat"],
                    color=my_palette[cond], s=40, zorder=3
                )

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

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    events, intervals = load_data(args.events, args.intervals)
    intervals = restrict_intervals_to_events(events, intervals)
    reference, treatments, order = get_condition_order(events)

    sd_df = assign_sd_to_activation(events, intervals, args.post_light_window)
    filtered_sd_df = apply_sd_filter(sd_df, args.sd_filter)

    activation_summary = compute_activation_summary(intervals)
    summary = compute_metrics(filtered_sd_df, activation_summary)

    base_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    my_palette = {reference: "tab:blue"}
    for cond, color in zip(treatments, base_colors):
        my_palette[cond] = color

    recording_offsets = {
        rec: np.random.uniform(-0.05, 0.05)
        for rec in summary["Recording"].unique()
    }

    plot_metrics(
        summary, reference, treatments, order, my_palette,
        recording_offsets, args.wilcoxon, args.sd_filter
    )


if __name__ == "__main__":
    main()