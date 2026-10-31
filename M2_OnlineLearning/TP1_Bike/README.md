# TP1 — UCI Bike Sharing

> This folder contains the exercise guide, Python requirements and dataset.
> The solution notebook described below is not included in this repository.

## Solution notebook (not included)

**Online_Learning_UCI_Bike_Sharing_Complete_Solution.ipynb**

An entirely English, fully solved practical accompanying Paul Minchella's
[course slides](../Cours_Online_Learning.pdf). It contains 18 exercises (0–17), worked
mathematical answers, executable implementations, result tables and eight figures.
Figures use a generic six-color palette, with distinct colors and line styles for each method.

The notebook follows the requested structure:

- **Part 0:** useful packages, configuration and reproducibility.
- **Part I:** descriptive statistics, availability of information, strictly causal features and chronological partitioning.
- **Part II:** FTL, full-loss FTRL, projected OGD, constrained hindsight oracles and selection by minimum validation regret per day.
- **Part III:** final test metrics, error and stability diagnostics, a finite-time bound and limits of static average regret.

## Included files

```text
Guide_Exercices_1_12_FR.pdf
README.md
requirements.txt
online_learning_bike_sharing_data/
    day.csv
    Readme.txt
    bike_sharing.zip
    PROVENANCE.md
```

Executing the notebook also creates `online_learning_bike_sharing_results/`, with
indexed interval bounds (CSV and NPZ), axis labels, metric scorecards and a JSON
manifest documenting the resampling settings. The original course PDF and previous
notebooks are unchanged.

## Run the solution notebook (when available)

Use Python 3.10 or newer (execution was tested on Python 3.12).
Place the solution notebook in `TP1_Bike/`, alongside the data folder.
From the repository root, run:

```powershell
cd M2_OnlineLearning/TP1_Bike
python -m pip install -r requirements.txt
# Only if you need a Jupyter interface:
python -m pip install jupyterlab
python -m jupyterlab
```

Open the notebook, select the intended Python kernel, then **Restart Kernel and Run All**.
VS Code or another compatible Jupyter interface can also open the notebook.
Run from the notebook's directory so the relative data folder is found.

The original solution notebook contains executed outputs. The verified local CSV permits
an offline rerun. If the CSV is missing, the loader uses the bundled archive or downloads
the original official archive. Changed data fail checksum validation rather than being
silently accepted. Exact executed package versions are printed in Part 0; the requirements
specify compatible minimum versions, not a bit-for-bit environment lock.

## Dataset and protocol

UCI **Bike Sharing**, Hadi Fanaee-T, https://doi.org/10.24432/C5W894.
Daily Capital Bikeshare rental counts, Washington, D.C., 2011–2012.
License on the current UCI page: **CC BY 4.0**. Full attribution, the associated
Fanaee-T and Gama paper, source links and checksums are in the notebook and
`online_learning_bike_sharing_data/PROVENANCE.md`.

Of 731 raw days, 724 are usable after lag construction:

| Period | Days | Dates |
|:--|--:|:--|
| Calibration | 90 | 2011-01-08 to 2011-04-07 |
| Validation | 180 | 2011-04-08 to 2011-10-04 |
| Test | 454 | 2011-10-05 to 2012-12-31 |

All scored predictions precede their outcomes. Realized same-day weather and the
components of today's target are excluded from predictors. No full-data fitted scaler
or shuffled split is used. The algorithms and hindsight comparator share the same
unregularized half squared loss and coefficient ball of radius 4.

Hyperparameters minimize final validation static regret per validation day, which is
equivalent to minimizing validation mean loss with a common comparator. The selected
FTRL penalty is **0.01** and the OGD scale is **3**, with step `3 / sqrt(k)`.
The OGD clock starts after calibration and continues through validation and testing.
During testing coefficients continue to learn, but hyperparameters remain frozen.

## Expected results

Approximate test RMSE, in rentals per day:

| Predictor | RMSE | Empirical 95% interval |
|:--|--:|--:|
| FTL | 1,047.6 | [933.0, 1,177.8] |
| FTRL | 1,050.6 | [937.9, 1,178.4] |
| OGD | 1,076.1 | [956.2, 1,213.2] |
| Previous day | 1,193.0 | [1,047.8, 1,333.6] |
| Previous week | 1,510.3 | [1,315.8, 1,703.6] |

OGD reduces RMSE by about **9.8%** relative to yesterday's count and **28.8%** relative
to last week's count. It is competitive here, not the winner of every comparison.
FTRL is the validation-selected primary method; FTL has the best realized test score.
A retrospective test winner is not treated as a method chosen in advance.

The notebook distinguishes normalized loss/regret from errors in original rental units.
It includes tail errors, recent-window RMSE, an inaccurate zero-static-regret example,
an all-zero counterexample to a universal guarantee, and a stronger fixed 13-week
blockwise hindsight benchmark. The latter is diagnostic,
not a deployable competitor or a consequence of the static-regret theorem.

## Empirical 95% intervals

The notebook uses 1,000 paired circular moving-block bootstrap samples, with
seven-day blocks and a fixed seed. The 2.5th and 97.5th percentiles are saved before
plotting; curve bands use opacity `0.1`. Each prefix is resampled using only its
own history. Bounds before 28 observations are `NaN` (insufficient declared history).
The same draws are used for competing methods, and the unregularized constrained
hindsight comparator is refitted in every draw used to compute regret.

Intervals cover validation-grid metrics, loss/regret prefix curves, all final-test
scorecard metrics, descriptive summaries, rolling RMSE, mean state diagnostics and
tracking comparisons. Timing intervals instead bootstrap 15 repeated fresh runs.
Raw observations, individual forecasts/errors, exact counts and theoretical bounds
are not assigned artificial sampling intervals. Maximum-error bootstrap ranges do
not quantify unseen extremes.

The RMSE bar chart immediately after the cost table pools all **634 validation and
test forecast days**, excluding calibration. This is pooled RMSE, not an average of
window RMSEs. Since validation participated in selection, this pooled chart is
descriptive, not an independent holdout; the final-test scorecard remains separate.

These are approximate, conditional, **pointwise** empirical intervals, not simultaneous
bands, confidence sequences or proven 95% coverage. Recorded forecast paths are held
fixed: learners are not retrained and hyperparameters are not reselected. Weekly
blocks only approximate temporal dependence; seasonal drift can invalidate coverage.

Example of a saved interval after `n` test forecasts:

```python
test_curve_ci["OGD"]["average_regret"][n - 1]  # [lower_95, upper_95]
```

## Validation and scope

The notebook was executed end-to-end; all code cells completed and all built-in
assertions passed. A second execution from a separate folder with network access
explicitly blocked reproduced the metric outputs using only the bundled CSV. Checks cover feature causality, finite-difference gradients,
constrained quadratic fitting against an independent solver, zero-design and singular
cases, active/inactive constraints, anchored ridge, KKT residuals, predict-before-update,
FTL/FTRL equivalence at zero penalty, correct prefix hindsight loss and uninterrupted
versus split evaluation. Additional checks cover paired weighted bootstrap optima,
active/singular cases, interval ordering and shape, prefix scaling, grid alignment,
feasible-baseline comparisons and safe NPZ reloading without pickle. The offline
rerun also reproduces all non-timing interval arrays exactly. Predictions and seeded
resampling are deterministic up to floating-point precision; runtime measurements
are hardware dependent.

This is a pedagogically selected historical stream, not a preregistered benchmark.
Small regret relative to one fixed vector does not establish small absolute errors,
good tracking, parameter convergence, statistical significance or operational optimality.
