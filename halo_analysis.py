"""
Light-Activation-Normalized SD Analysis
========================================

Analyzes spreading depolarization (SD) events relative to Halorhodopsin
light stimulation, using the light activation itself (rather than raw
recording duration) as the normalization unit.

All the actual logic lives in sd_functions.py; this file is just the
CLI entry point. See power_analysis.py for a companion script that
estimates sample sizes needed to detect effects seen in a pilot run.

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
"""

import argparse

import numpy as np

from sd_functions import (
    build_summary,
    get_condition_order,
    print_metric_summary,
    save_summary_csv,
    plot_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Light-activation-normalized SD analysis.")
    parser.add_argument("--events", required=True, help="Path to SD-event CSV.")
    parser.add_argument("--intervals", required=True, help="Path to light-activation interval CSV.")
    parser.add_argument("--save-csv", action="store_true",
                         help="Prompt for a location and save the analysis summary as a CSV.")
    parser.add_argument("--wilcoxon", action="store_true",
                         help="Use Wilcoxon signed-rank test instead of paired t-test.")
    parser.add_argument("--post-light-window", type=float, default=30,
                         help="Seconds after light turns off to still classify an SD as induced.")
    parser.add_argument("--sd-filter", choices=["induced", "spontaneous", "all"], default="induced",
                         help="Which SDs feed the metrics: induced, spontaneous, or all.")
    parser.add_argument("--no-plots", action="store_true",
                         help="Print results without opening the matplotlib figures.")
    return parser.parse_args()


def main():
    args = parse_args()

    events, intervals, summary = build_summary(
        args.events, args.intervals, args.post_light_window, args.sd_filter
    )
    reference, treatments, order = get_condition_order(events)

    print_metric_summary(summary, reference, treatments, order, args.wilcoxon)

    base_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    my_palette = {reference: "tab:blue"}
    for cond, color in zip(treatments, base_colors):
        my_palette[cond] = color

    recording_offsets = {
        rec: np.random.uniform(-0.1, 0.1) for rec in summary["Recording"].unique()
    }

    if args.save_csv:
        save_summary_csv(summary)

    if not args.no_plots:
        plot_metrics(summary, reference, treatments, order, my_palette,
                     recording_offsets, args.wilcoxon, args.sd_filter)


if __name__ == "__main__":
    main()
