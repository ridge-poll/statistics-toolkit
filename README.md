# Experimental Event Analysis Toolkit

A Python toolbox for analyzing event-based experimental recordings and generating publication-quality visualizations.

This repository supports workflows for:
- condition-level event rate analysis
- paired comparisons within recordings
- control-normalized (delta) analysis
- statistical testing and visualization of experimental conditions
- statistical power and sample-size estimation

Designed for electrophysiology and imaging datasets where events are sparse and structured across recording epochs.

---

## Analysis Pipelines

### 1. Condition-Level Event Analysis (`paired_condition_analysis.py`)

This script performs within-dataset analysis across conditions.

**Key functions:**
- Computes event-derived metrics per recording and condition
- Calculates:
  - Event rate (events/sec)
  - Amplitude
  - Duration
  - Area under curve (AUC)
- Performs paired statistical comparisons vs a reference condition
- Produces raw-value visualizations with:
  - mean ± SEM
  - individual data points
  - significance annotations

---

#### Optional: Wilcoxon Signed-Rank Test

By default, the analysis scripts use paired Student's *t*-tests for statistical comparisons.

For nonparametric paired analysis, either script can instead perform a two-sided Wilcoxon signed-rank test:

```bash
python paired_condition_analysis.py --wilcoxon
```

or

```bash
python delta_analysis.py --wilcoxon
```

The `--wilcoxon` flag changes only the statistical hypothesis test. All summary metrics, paired-recording matching, plots, and visualizations remain unaltered.

The power analysis script currently uses paired *t*-test power calculations and does not support Wilcoxon power estimation.

### 2. Paired Delta Analysis (`delta_analysis.py`)

This script computes control-normalized differences within each recording.

**Key functions:**
- Computes within-recording differences relative to control:
  - Δ Rate
  - Δ Amplitude
  - Δ Duration
  - Δ AUC
- Performs one-sample statistical tests against zero
- Focuses on effect size rather than absolute values
- Produces clearer cross-condition comparisons
- Includes baseline reference line at zero

---

### 3. Statistical Power Analysis (`power_analysis.py`)

This script estimates statistical power and required sample size for paired experiments.

**Key functions:**
- Uses the same paired-recording framework as the analysis scripts
- Computes paired treatment–control differences for:
  - Rate
  - Amplitude
  - Duration
  - AUC
- Calculates:
  - Current p-value
  - Estimated statistical power
  - Total recordings required to reach target power
  - Additional recordings needed
- Uses paired t-test power calculations (`statsmodels`)
- Produces:
  - color-coded summary tables
  - power-vs-sample-size curves

---

## Features

- Interactive CSV file loading (file picker GUI)
- Automatic grouping by:
  - Recording
  - Condition
- Handles zero-event recordings explicitly when valid windows exist
- Derived metrics:
  - Event rate
  - Optional AUC normalization
- Statistical testing:
  - Paired t-tests (default)
  - Wilcoxon signed-rank tests (`--wilcoxon` option)
  - One-sample t-tests (delta analysis)
- Statistical power analysis:
  - achieved power estimation
  - required sample-size estimation
- Publication-style plots:
  - bar plots with SEM
  - jittered raw data overlays
  - consistent condition color mapping
  - significance annotations

---

## Example Output

<img src="assets/example_spreadsheet.png" width="500" height="250">

---

## Installation

```bash
pip install pandas numpy matplotlib scipy statsmodels
```

## Usage

### Condition-level Analysis

```bash
python paired_condition_analysis.py
```
This script:
- Loads a CSV file via file picker
- Computes summary metrics per recording and condition
- Performs paired statistical comparisons against a reference condition
- Generates multi-panel plots for all metrics


### Paired Delta Analysis

```bash
python paired_delta_analysis.py
```
This script:
- Loads a CSV file via file picker
- Converts all metrics into Δ (difference-from-control) values
- Performs one-sample t-tests against zero
- Produces effect-size centered visualizations

*In effect, these two scripts produce the same p-values. The first compares a single treatment to the control, the second provides visualization for multiple treatments compared to their respective paired control.*

### Power Analysis

```bash
python power_analysis.py
```

This script:
- Loads a CSV file via file picker
- Computes paired treatment–control differences
- Estimates current statistical power
- Calculates total and additional recordings needed to reach target power
- Produces:
  - console summaries
  - color-coded result tables
  - power curves for each metric

Default settings:
- α = 0.05
- target power = 0.80
- preferred reference condition = `ACSF`

If `ACSF` is absent, the script automatically selects the condition with the greatest paired coverage and reports this choice.

## Input Data Format
Input CSV files must contain event-level data with the following categories:

| Recording | Condition | Start_s | End_s | MaxAmp1 | Duration1 | AUC1 |
|----------|----------|--------|-------|--------|----------|------|
| R1a | ACSF | 1200 | 2000 | 4.67 | 194.54 | 694.75 |
| R1a | ACSF | 1200 | 2000 | 3.14 | 119.18 | 273.35 |
| R1a | ACSF | 1200 | 2000 | 2.33 | 83.72 | 142.24 |
| R1a | H-40 | 2000 | 2800 | 3.04 | 49.54 | 92.43 |
| R1a | H-40 | 2000 | 2800 | 5.25 | 128.40 | 463.19 |
| R2a | ACSF | 1230 | 1736 | 2.83 | 88.96 | 141.08 |
| R2a | H-40 | 1850 | 2400 | 2.04 | 22.24 | 26.33 |
| R3a | ACSF | 1200 | 2000 | 2.14 | 94.62 | 118.76 |
| R3a | H-40 | 2000 | 2800 | 3.18 | 121.34 | 212.47 |
| R4a | ACSF | 1000 | 1600 | 2.48 | 111.27 | 151.23 |
| R4a | H-40 | 1600 | 2200 | 3.02 | 131.42 | 218.66 |
| R1b | ACSF | 0 | 480 | 10.43 | 3.00 | 13.38 |
| R1b | Mannitol | 480 | 1000 | 22.19 | 29.78 | 380.54 |
| R2b | ACSF | 0 | 840 | 39.31 | 245.08 | 6350.23 |
| R2b | Mannitol | 840 | 1500 | 33.35 | 86.56 | 1470.91 |
| R3b | ACSF | 0 | 600 | 30.73 | 36.68 | 566.57 |
| R3b | Mannitol | 600 | 1473 | 20.57 | 163.04 | 1720.73 |
| R4b | ACSF | 0 | 780 | 147.30 | 163.84 | 13799.28 |
| R4b | Mannitol | 780 | 1890 | NaN | NaN | NaN |
| R5b | ACSF | 660 | 1350 | 7.28 | 181.52 | 901.82 |
| R5b | Mannitol | 0 | 660 | NaN | NaN | NaN |

Notes
- Recordings must have unique identifiers so conditions can be paired correctly.
- Zero-event conditions should still be included if a recording window exists.
- In these cases, include:
  - valid Start_s and End_s
  - NaN for event-derived values (e.g., amplitude, duration, AUC)
- Event rate is computed using non-NaN event rows and recording-window duration (`End_s − Start_s`).