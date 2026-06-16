import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from scipy.stats import ttest_rel, ttest_1samp
import tkinter as tk
from tkinter import filedialog

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

# Summarize data
summary = df.groupby(["Recording", "Condition"]).agg(
    Amplitude=("MaxAmp1", "mean"),
    Duration=("Duration1", "mean"),
    ShiftSlope=("ShiftSlope1", "mean"),
    AUC=("AUC1", "mean"),
    EventCount=("Duration1", "count"),
    Start=("Start_s", "first"),
    End=("End_s", "first")
).reset_index()


# Fix event count (missing -> 0 events)
summary["EventCount"] = summary["EventCount"].fillna(0)

# Find rate -> events over time 
duration = summary["End"] - summary["Start"]
summary["Rate"] = np.where((summary["EventCount"] > 0) &
                           (duration > 0),summary["EventCount"] / duration,0)

# Find normalized AUC -> AUC over amplitude (currently not wanted by Kojo)
summary["NormAUC"] = summary["AUC"] / summary["Amplitude"]


# Build delta frames
delta_rows = []

for rec in summary["Recording"].unique():

    rec_df = summary[summary["Recording"] == rec]

    control_df = rec_df[rec_df["Condition"] == reference]
    if control_df.empty:
        continue

    control = control_df.iloc[0]

    for _, row in rec_df.iterrows():
        cond = row["Condition"]
        if cond == reference:
            continue

        delta_rows.append({
            "Recording": rec,
            "Condition": cond,
            "Rate": row["Rate"] - control["Rate"],
            "Amplitude": row["Amplitude"] - control["Amplitude"],
            "Duration": row["Duration"] - control["Duration"],
            "AUC": row["AUC"] - control["AUC"]
        })

delta_df = pd.DataFrame(delta_rows)



# Reshape for plotting
delta_long = delta_df.melt(
    id_vars=["Recording", "Condition"],
    var_name="Metric",
    value_name="Value"
)


# Colors
base_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
my_palette = {}

for cond, color in zip(treatments, base_colors):
    my_palette[cond] = color


# Plot settings
metrics = ["Rate", "Amplitude", "Duration", "AUC"]
labels = [
    "Δ events/sec",
    "Δ mV",
    "Δ sec",
    "Δ mV·sec"
]


fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))

for ax, m, lbl in zip(axes, metrics, labels):

    subset = delta_long[delta_long["Metric"] == m]

    means = []
    sems = []
    all_values = []

    for cond in treatments:
        values = subset[subset["Condition"] == cond]["Value"].dropna().values
        all_values.append(values)

        if len(values) > 0:
            means.append(np.mean(values))
            sems.append(np.std(values, ddof=1) / np.sqrt(len(values)))
        else:
            means.append(np.nan)
            sems.append(np.nan)

    # Bars
    for i, cond in enumerate(treatments):
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
        range(len(treatments)),
        means,
        yerr=sems,
        fmt='none',
        capsize=5,
        color='black'
    )

    # Scatter
    for i, values in enumerate(all_values):
        jitter = np.random.normal(i, 0.05, size=len(values))
        ax.scatter(
            jitter,
            values,
            color=my_palette[treatments[i]],
            s=40,
            zorder=3
        )

    # Stats (one-sample vs 0)
    p_text = []

    for cond in treatments:
        values = subset[subset["Condition"] == cond]["Value"].dropna()

        if len(values) > 1:
            _, pval = ttest_1samp(values, 0)
            p_text.append(f"{cond}: p={pval:.3f}")
        else:
            p_text.append(f"{cond}: n<2")

    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(treatments)
    ax.set_ylabel(lbl)
    ax.set_title(
        f"{m}\n" + "\n".join(p_text),
        fontsize=10,
        fontweight="bold"
    )

    # Reference line (no effect)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle("SD Properties in Mice\n(Δ from Control)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()