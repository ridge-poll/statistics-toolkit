import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import tkinter as tk
from tkinter import filedialog
import argparse

# ----- Flags and such -----
parser = argparse.ArgumentParser()
parser.add_argument(
    "--wilcoxon",
    action="store_true",
    help="Use Wilcoxon signed-rank test instead of paired t-test."
)

parser.add_argument(
    "--post-light-window",
    type=float,
    default=30,
    help="Time in seconds after light turns off to still classify an SD as induced."
)

args = parser.parse_args()


# ----- File picker -----
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select CSV file",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
df = pd.read_csv(file_path)
# -----------------------


# Define condition structure
reference = df["Condition"].iloc[0]
conditions = list(pd.unique(df["Condition"]))
treatments = [c for c in conditions if c != reference]
order = [reference] + treatments


summary = df.groupby(["Recording", "Condition"]).agg(
    Amplitude=("MaxAmp1", "mean"),
    Duration=("Duration1", "mean"),
    ShiftSlope=("ShiftSlope1", "mean"),
    AUC=("AUC1", "mean"),
    EventCount=("Duration1", "count"),
    Start=("Start_s", "first"),
    End=("End_s", "first")
).reset_index()



# Find rate -> events over time 
duration = summary["End"] - summary["Start"]
summary["Rate"] = np.where((summary["EventCount"] > 0) &
                           (duration > 0),summary["EventCount"] / duration,0)

# Find normalized AUC -> AUC over amplitude (currently not wanted by Kojo)
summary["NormAUC"] = summary["AUC"] / summary["Amplitude"]


# Reshape for plotting
long_df = summary.melt(
    id_vars=["Recording", "Condition"],
    value_vars=["Amplitude", "Duration", "AUC", "Rate"],
    var_name="Metric",
    value_name="Value"
)


# color palette
base_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
my_palette = {reference: "tab:blue"}

for cond, color in zip(treatments, base_colors):
    my_palette[cond] = color


# Plotting
metrics = ["Rate", "Amplitude", "Duration", "AUC"]
labels = [
    "events/sec",
    "mV",
    "sec",
    "mV·sec"
]

# Give each recording a fixed horizontal jitter
recording_offsets = {
    rec: np.random.uniform(-0.05, 0.05)
    for rec in summary["Recording"].unique()
}

fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
fig.suptitle(f"Cross-Species SDs in KCC2 Blocker: {cond}", fontsize=14, fontweight="bold")

for ax, m, lbl in zip(axes, metrics, labels):
    subset = long_df[long_df["Metric"] == m]

    means = []
    sems = []
    all_values = []

    for cond in order:
        values = subset[subset["Condition"] == cond]["Value"].dropna().values
        all_values.append(values)

        if len(values) > 0:
            means.append(np.mean(values))
            sems.append(np.std(values, ddof=1) / np.sqrt(len(values)))
        else:
            means.append(0)
            sems.append(0)

    # Bars
    for i, cond in enumerate(order):
        ax.bar(
            i,
            means[i],
            width=0.4,
            color=my_palette[cond],
            alpha=0.25,
            edgecolor=my_palette[cond],
            linewidth=2
        )

    # Error bars
    ax.errorbar(
        range(len(order)),
        means,
        yerr=sems,
        fmt='none',
        capsize=5,
        color='black'
    )

    # Draw paired lines
    for cond in treatments:

        control = subset[subset["Condition"] == reference][["Recording", "Value"]]
        treat = subset[subset["Condition"] == cond][["Recording", "Value"]]

        merged = pd.merge(
            control,
            treat,
            on="Recording",
            suffixes=("_control", "_treat")
        ).dropna()

        x0 = order.index(reference)
        x1 = order.index(cond)

        for _, row in merged.iterrows():
            dx = recording_offsets[row["Recording"]]

            ax.plot(
                [x0 + dx, x1 + dx],
                [row["Value_control"], row["Value_treat"]],
                color="gray",
                linewidth=1,
                alpha=0.6,
                zorder=2
            )

    # Scatter (raw data)
    for i, cond in enumerate(order):

        cond_df = subset[subset["Condition"] == cond]

        x = [
            i + recording_offsets[r]
            for r in cond_df["Recording"]
        ]

        ax.scatter(
            x,
            cond_df["Value"],
            color=my_palette[cond],
            s=40,
            zorder=3
        )

    # Stats
    p_text = []
    control_df = subset[subset["Condition"] == reference][["Recording", "Value"]]

    for cond in treatments:
        group_df = subset[subset["Condition"] == cond][["Recording", "Value"]]

        merged = pd.merge(
            control_df,
            group_df,
            on="Recording",
            suffixes=("_control", "_treat")
        ).dropna()

        if len(merged) > 1:
            if args.wilcoxon:
                try:
                    _, pval = wilcoxon(
                        merged["Value_control"],
                        merged["Value_treat"],
                        zero_method="wilcox",
                        alternative="two-sided"
                    )
                except ValueError:
                    p_text.append("all equal")
                    continue
            else:
                _, pval = ttest_rel(
                    merged["Value_control"],
                    merged["Value_treat"],
                    nan_policy="omit"
                )

            p_text.append(f"p={pval:.6f}")
        else:
            p_text.append("n<2")

    # --- Formatting ---
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(lbl)
    ax.set_title(
    f"{m}\n" + "\n".join(p_text),
    fontsize=10,
    fontweight="bold"
)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


################ New ################


# ============================================================
# HALORHODOPSIN LIGHT ANALYSIS
# ============================================================

# Only rows with an actual SD are included in the light analysis.
# Rows with NaN MaxAmp1 are bookkeeping rows representing no SD.
light_df = df[df["MaxAmp1"].notna()].copy()

# Make sure timing columns are numeric
light_df["Latency1"] = pd.to_numeric(
    light_df["Latency1"],
    errors="coerce"
)

light_df["Change_Time"] = pd.to_numeric(
    light_df["Change_Time"],
    errors="coerce"
)

# Time between the event and the most recent light change
light_df["Time_From_Change"] = (
    light_df["Latency1"] - light_df["Change_Time"]
)

# ------------------------------------------------------------
# Classify SDs as induced or spontaneous
# ------------------------------------------------------------

light_df["SD_Type"] = "spontaneous"

# Light is currently ON
light_df.loc[
    light_df["Light_Status"].str.lower() == "on",
    "SD_Type"
] = "induced"

# Light has just turned OFF:
# still classify as induced if within the post-light window
light_df.loc[
    (light_df["Light_Status"].str.lower() == "off") &
    (light_df["Time_From_Change"] >= 0) &
    (light_df["Time_From_Change"] <= args.post_light_window),
    "SD_Type"
] = "induced"

# ------------------------------------------------------------
# Count total, induced, and spontaneous SDs per recording
# ------------------------------------------------------------

light_counts = light_df.groupby(
    ["Recording", "Condition"]
).agg(
    Total_SDs=("MaxAmp1", "count"),
    Induced_SDs=(
        "SD_Type",
        lambda x: (x == "induced").sum()
    ),
    Spontaneous_SDs=(
        "SD_Type",
        lambda x: (x == "spontaneous").sum()
    )
).reset_index()

# ------------------------------------------------------------
# Find first SD after each light activation
# ------------------------------------------------------------

# Identify rows where the light was turned ON.
#
# Each unique Change_Time represents a light activation.

light_on = light_df[
    (light_df["Light_Status"].str.lower() == "on") &
    light_df["Latency1"].notna() &
    light_df["Change_Time"].notna()
].copy()

light_on["Latency_From_Light_On"] = (
    light_on["Latency1"] - light_on["Change_Time"]
)

# Only SDs occurring after the light was turned on
light_on = light_on[
    light_on["Latency_From_Light_On"] >= 0
]

# First SD after each light activation
first_sd = (
    light_on
    .sort_values("Latency_From_Light_On")
    .groupby(
        ["Recording", "Condition", "Change_Time"],
        as_index=False
    )
    .first()
)

# One value per recording:
# mean latency of the first SD following each activation

latency_summary = first_sd.groupby(
    ["Recording", "Condition"]
).agg(
    First_SD_Latency=("Latency_From_Light_On", "mean")
).reset_index()


# ============================================================
# COMBINED FOUR-PANEL LIGHT ANALYSIS FIGURE
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes = axes.flatten()

panel_specs = [
    (
        light_counts,
        "Total_SDs",
        "Total SDs per Recording",
        "SD count"
    ),
    (
        light_counts,
        "Induced_SDs",
        "Induced SDs per Recording",
        "Induced SD count"
    ),
    (
        light_counts,
        "Spontaneous_SDs",
        "Spontaneous SDs per Recording",
        "Spontaneous SD count"
    ),
    (
        latency_summary,
        "First_SD_Latency",
        "Time From Light Activation to First SD",
        "Latency (sec)"
    )
]

for ax, (data, value_column, title, ylabel) in zip(
    axes,
    panel_specs
):

    # --- compute means and SEMs ---
    means = []
    sems = []

    for cond in order:

        values = data[
            data["Condition"] == cond
        ][value_column].dropna().values

        if len(values) > 0:

            means.append(np.mean(values))

            if len(values) > 1:
                sems.append(
                    np.std(values, ddof=1) /
                    np.sqrt(len(values))
                )
            else:
                sems.append(0)

        else:
            means.append(np.nan)
            sems.append(np.nan)

    # --- bars ---
    for i, cond in enumerate(order):

        ax.bar(
            i,
            means[i],
            width=0.4,
            color=my_palette[cond],
            alpha=0.25,
            edgecolor=my_palette[cond],
            linewidth=2
        )

    # --- error bars ---
    ax.errorbar(
        range(len(order)),
        means,
        yerr=sems,
        fmt="none",
        capsize=5,
        color="black"
    )

    # --- paired lines + scatter ---

    p_text = "n<2"

    control = data[
        data["Condition"] == reference
    ][["Recording", value_column]].dropna()

    for cond in treatments:

        treat = data[
            data["Condition"] == cond
        ][["Recording", value_column]].dropna()

        merged = pd.merge(
            control,
            treat,
            on="Recording",
            suffixes=("_control", "_treat")
        ).dropna()

        x0 = order.index(reference)
        x1 = order.index(cond)

        for _, row in merged.iterrows():

            dx = recording_offsets.get(
                row["Recording"],
                0
            )

            ax.plot(
                [x0 + dx, x1 + dx],
                [
                    row[f"{value_column}_control"],
                    row[f"{value_column}_treat"]
                ],
                color="gray",
                linewidth=1,
                alpha=0.6,
                zorder=2
            )

            ax.scatter(
                x0 + dx,
                row[f"{value_column}_control"],
                color=my_palette[reference],
                s=40,
                zorder=3
            )

            ax.scatter(
                x1 + dx,
                row[f"{value_column}_treat"],
                color=my_palette[cond],
                s=40,
                zorder=3
            )

        # --- statistics ---

        if len(merged) > 1:

            if args.wilcoxon:

                try:
                    _, pval = wilcoxon(
                        merged[f"{value_column}_control"],
                        merged[f"{value_column}_treat"],
                        zero_method="wilcox",
                        alternative="two-sided"
                    )

                except ValueError:
                    pval = np.nan

            else:

                _, pval = ttest_rel(
                    merged[f"{value_column}_control"],
                    merged[f"{value_column}_treat"],
                    nan_policy="omit"
                )

            p_text = (
                "all equal"
                if np.isnan(pval)
                else f"p={pval:.6f}"
            )

    # --- formatting ---

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(ylabel)

    ax.set_title(
        f"{title}\n{p_text}",
        fontsize=10,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()