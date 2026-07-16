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

fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
fig.suptitle(f"Zebrafish SDs in Hypoosmotic Solution: {cond}", fontsize=14, fontweight="bold")

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

    # Scatter (raw data)
    for i, values in enumerate(all_values):
        jitter = np.random.normal(i, 0.05, size=len(values))
        ax.scatter(
            jitter,
            values,
            color=my_palette[order[i]],
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

            p_text.append(f"p={pval:.3f}")
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