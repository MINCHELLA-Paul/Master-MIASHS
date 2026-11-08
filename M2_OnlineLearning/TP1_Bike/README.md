# TP1 — UCI Bike Sharing

> This folder contains the English and French student notebooks, the Python requirements and the dataset.
> The solution notebooks described below are not included.

## Student notebooks

- [TP1_Bike_template_ENG.ipynb](TP1_Bike_template_ENG.ipynb) — English edition.
- [TP1_Bike_template_FR.ipynb](TP1_Bike_template_FR.ipynb) — Version française.

Complete the function and class bodies marked `# To complete` / `# À compléter`, and write your answers in the
**Your answer:** / **Votre réponse :** cells. Function names, signatures and docstrings are mandatory, because later
cells call them. Both editions have the same structure (59 cells), interfaces and data; each edition is written
entirely in its own language, docstrings included. These are unfinished student templates, not executed solutions:
complete the cells in order before attempting a full run.

## Solution notebooks (not included)

**TP1_Bike_solutions_ENG.ipynb** and **TP1_Bike_solutions_FR.ipynb**

Fully solved editions accompanying Paul Minchella's [course slides](../Cours_Online_Learning.pdf), with solution code,
short written answers and seven figures. Figures use the six-colour course palette (teal, sky blue, blue, dark blue,
orange, purple), with a distinct colour and line style for each model.

## Structure

- **Part 0:** packages, colour palette and data loading.
- **Part 1:** descriptive statistics — first rows and shape, summary table (mean, std, min, median, max, missing values,
  observations), plot of the observed series, and a table describing every column of `day.csv`.
- **Part 2:** online learning.
  - Protocol, regret, and the two model families: squared loss (OGD, OGD + Ridge) and newsvendor loss.
  - Question 2.1: checking the assumptions of the regret guarantees (compact domain, bounded covariates and target, convex losses).
  - Question 2.2: strictly causal covariates $\mathbf x_t = (1, \mathbf z_t, y_{t-1}, \dots, y_{t-7})$.
  - Question 2.3: train-then-test protocol, predicting before learning, with a plot of the two windows.
  - Implementation of each model, an empirical check that the optimal newsvendor order is the critical quantile
    $\tau = c_b/(c_b + c_h)$, and two naive baselines.
- **Part 3:** final metrics — scorecard, RMSE and newsvendor-cost bar charts, average regret over time, conclusion.

## Included files

```text
TP1_Bike_template_ENG.ipynb
TP1_Bike_template_FR.ipynb
README.md
requirements.txt
online_learning_bike_sharing_data/
    day.csv
    Readme.txt
    bike_sharing.zip
    PROVENANCE.md
```

## Open the student notebooks

Use Python 3.10 or newer. The notebooks need NumPy, pandas, Matplotlib, SciPy and a Jupyter interface.
From the repository root, run:

```powershell
cd M2_OnlineLearning/TP1_Bike
python -m pip install -r requirements.txt
# Only if you need a Jupyter interface:
python -m pip install jupyterlab
python -m jupyterlab TP1_Bike_template_FR.ipynb
```

For the English edition, open `TP1_Bike_template_ENG.ipynb` instead. VS Code or another compatible Jupyter interface
can also open the notebooks. Run from the notebook's directory so the relative data folder is found.

The loader reads `online_learning_bike_sharing_data/day.csv` when it is present, so the notebooks run offline.
If the CSV is missing, it downloads the official UCI archive, extracts `day.csv` only and caches it in that folder.

## Dataset

UCI **Bike Sharing**, Hadi Fanaee-T, https://doi.org/10.24432/C5W894.
Daily Capital Bikeshare rental counts, Washington, D.C., 2011–2012.
License on the current UCI page: **CC BY 4.0**. Please also cite Hadi Fanaee-T and João Gama (2013),
*Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence.
Source links are in `online_learning_bike_sharing_data/PROVENANCE.md`.

## Protocol and modelling choices

Of 731 raw days, 724 are usable after building seven demand lags:

| Window | Days | Dates | Role |
|:--|--:|:--|:--|
| Train | 358 | 2011-01-08 to 2011-12-31 | online learning and hyperparameter selection |
| Test | 366 | 2012-01-01 to 2012-12-31 | evaluation (predict, then learn) |

- **Target:** $y_t = \texttt{cnt}_t / 10{,}000 \in [0, 1]$. The scale is declared a priori, not estimated.
- **Covariates (d = 12):** intercept, `workingday`, `holiday`, a yearly cycle ($\sin$, $\cos$ of the day of the year)
  and the lags $y_{t-1}, \dots, y_{t-7}$, hence $\mid\mathbf x_t\mid_2 \le R = \sqrt{10}$ by construction. Same-day weather,
  `casual`, `registered` and `instant` are excluded.
- **Feasible set:** the coefficient ball $\{\|\theta\|_2 \le 2\}$. The newsvendor order without covariates lives in
  $[0, Q_{\max}]$, with $Q_{\max} = 10{,}000$ bikes.
- **Newsvendor costs:** the dataset provides none, so we set $c_b = 3$ (shortage) and $c_h = 1$ (excess), i.e. $\tau = 0.75$.
- **Models:** projected OGD with a constant step; OGD + Ridge with the capped step $\eta_t = \min\{1/R^2,\ 1/(\lambda t)\}$;
  newsvendor OGD without covariates; contextual newsvendor OGD $q_t = \mathbf x_t^\top \theta_t$; naive baselines
  (yesterday's count, last week's count).
- **Selection:** each hyperparameter ($\eta$ or $\lambda$) minimises the online average loss on the training window
  (squared loss for the squared-loss models, newsvendor cost for the newsvendor models). The grids include the
  theoretical steps.
- **Test:** hyperparameters are frozen, but the models keep learning from their end-of-2011 state, without resetting
  their clock. Every test prediction is made before its outcome is revealed. No shuffled split and no full-data scaler is used.

## Metrics

- **Scorecard on 2012:** RMSE, MAE and bias in bikes per day, newsvendor cost per day and coverage $\mathbb{P}(y_t \le \widehat y_t)$.
- **Bar charts:** RMSE and newsvendor cost per model. The newsvendor models are expected **not** to have the lowest RMSE.
  They answer a different need ("how many bikes should be made available?", targeting the 0.75 quantile) from the
  squared-loss models ("how many bikes will be rented?", targeting the mean). Each family is judged with its own loss.
- **Average regret over time:** $R_n/n$ on every test prefix, each model against the best fixed comparator for its own loss.
  These comparators are constrained least squares (ridge-penalised for OGD + Ridge), the best constant order, or a linear
  quantile regression solved by linear programming.

## Built-in checks

The notebooks include assertions for the feature bounds $\|\mathbf x_t\|_2 \le R$ and $|y_t| \le 1$, the causality of the
covariates (changing one day's demand changes no covariate of that day or earlier), and the predict-before-update loop
(changing $y_t$ does not change the prediction for day $t$).

## Scope

This is a pedagogically selected historical stream, not a benchmark. Small regret relative to one fixed parameter
establishes neither small absolute errors nor good tracking of a drifting demand. On this growing series, an adaptive
policy can even have strongly negative regret against a fixed comparator.
