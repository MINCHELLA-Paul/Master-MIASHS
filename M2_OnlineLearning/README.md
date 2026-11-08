# M2 — Online Learning / Apprentissage en ligne

<p align="center">
  <img src="Logo_Oventi_ERIC_Lyon2.png" alt="Logos Oventi, ERIC et Université Lumière Lyon 2" width="600">
</p>

Supports pédagogiques d'**Online Learning** — **Master 2 MIASHS, Université Lumière Lyon 2**.

---

## Français

### Présentation du cours

L'**apprentissage en ligne** étudie les méthodes qui prennent des décisions successives et adaptent leurs paramètres à mesure que les observations deviennent disponibles. Contrairement à un entraînement réalisé une fois sur un jeu de données figé, le protocole est séquentiel : **prédire ou décider → observer → subir une perte → mettre à jour le modèle**.

Le cours introduit notamment **Follow the Leader (FTL)**, **Follow the Regularized Leader (FTRL)** et **Online Gradient Descent (OGD)**. Il aborde les pertes convexes, la régularisation (dont Ridge), le regret cumulé et moyen, ainsi que les garanties théoriques associées. Les applications portent sur la prévision de demande et la prise de décision sous coûts asymétriques (*newsvendor*).

Pour une suite de pertes $\ell_1,\ldots,\ell_T$ et un ensemble de décisions $\mathcal K$, le **regret statique** s'écrit

$$ R_T = \sum_{t=1}^{T}\ell_t(\theta_t) - \min_{\theta\in\mathcal K}\sum_{t=1}^{T}\ell_t(\theta). $$

L'objectif est d'obtenir un regret moyen $R_T/T$ qui tend vers zéro, relativement à la meilleure décision **fixe** choisie a posteriori. Cela ne garantit pas, à lui seul, une faible erreur absolue de prévision.

### Ressources disponibles

| Support | Contenu |
|:--|:--|
| [Cours_Online_Learning.pdf](Cours_Online_Learning.pdf) | Diapositives du cours : protocole séquentiel, FTL, FTRL, OGD, regret et applications. |
| [Poly_Online_Learning.pdf](Poly_Online_Learning.pdf) | Polycopié de cours de Paul Minchella : définitions, développements mathématiques et exemples. |
| [Illustration_FTL_FTRL_OGD.ipynb](Illustration_FTL_FTRL_OGD.ipynb) | Illustration autonome de FTL, FTRL et OGD sur une demande synthétique, avec perte quadratique et regret moyen. |
| [TP1_Bike/README.md](TP1_Bike/README.md) | Présentation détaillée du TP1, de son protocole, des modèles et des données. |
| [TP1_Bike/TP1_Bike_template_FR.ipynb](TP1_Bike/TP1_Bike_template_FR.ipynb) | **TP1 — notebook étudiant en français**, à compléter (code et réponses rédigées). |
| [TP1_Bike/TP1_Bike_template_ENG.ipynb](TP1_Bike/TP1_Bike_template_ENG.ipynb) | **TP1 — student notebook in English**, to complete. |
| [TP1_Bike/requirements.txt](TP1_Bike/requirements.txt) | Dépendances Python pour le TP1. |
| [TP1_Bike/online_learning_bike_sharing_data/](TP1_Bike/online_learning_bike_sharing_data/) | Jeu de données UCI Bike Sharing et documentation de provenance. |

### TP1 — Bike Sharing : prévision et décision en ligne

Le TP1 utilise les **comptages quotidiens de locations de vélos à Washington, D.C.**, issus du jeu de données [UCI Bike Sharing](https://doi.org/10.24432/C5W894), pour étudier deux problèmes complémentaires :

1. **Prévoir la demande** avec une perte quadratique : OGD projeté et OGD avec régularisation Ridge.
2. **Décider d'une quantité à mettre à disposition** avec des coûts de manque et de surplus : OGD pour la perte *newsvendor*, sans puis avec covariables.

Les étudiants construisent des covariables disponibles avant la décision (calendrier et demandes passées), vérifient les hypothèses des garanties de regret, implémentent les mises à jour et comparent les approches à des références naïves. L'évaluation suit une séparation **chronologique** : apprentissage en 2011 puis évaluation en 2012, avec une prédiction réalisée **avant** la révélation de chaque demande journalière.

Le TP distingue les métriques de prévision (**RMSE, MAE, biais**), le coût opérationnel *newsvendor* et le **regret moyen**. Minimiser une erreur quadratique et minimiser un coût asymétrique ne correspondent pas nécessairement à la même décision : la seconde vise un quantile critique plutôt qu'une moyenne.

**À destination des étudiants :** les deux notebooks sont des **versions à compléter**, et non des corrigés exécutables directement. Respectez les noms et signatures des fonctions imposées et renseignez les cellules de réponses. Les notebooks de correction ne sont pas fournis dans ce dossier.

### Installation et exécution

**Prérequis :** Python **3.10+**, un environnement Jupyter (JupyterLab ou VS Code) et les dépendances du TP. Depuis un **répertoire de travail** où cloner le dépôt :

```bash
git clone https://github.com/MINCHELLA-Paul/Master-MIASHS.git
cd Master-MIASHS/M2_OnlineLearning
python -m pip install -r TP1_Bike/requirements.txt
python -m pip install jupyterlab
```

Pour ouvrir l'illustration FTL/FTRL/OGD :

```bash
python -m jupyterlab Illustration_FTL_FTRL_OGD.ipynb
```

Pour le TP1, lancez Jupyter **depuis le dossier du TP**, afin que les chemins relatifs vers les données soient corrects :

```bash
cd TP1_Bike
python -m jupyterlab TP1_Bike_template_FR.ipynb
# Ou, pour la version anglaise :
# python -m jupyterlab TP1_Bike_template_ENG.ipynb
```

Les données journalières sont incluses dans le sous-dossier du TP ; le notebook prévoit également leur téléchargement depuis l'UCI si le fichier CSV local est absent. Pour le détail des données et des consignes, consultez le [README du TP1](TP1_Bike/README.md).

**Parcours conseillé :** diapositives et polycopié → notebook illustratif FTL/FTRL/OGD → TP1 Bike Sharing (version française ou anglaise).

---

## English

### Course overview

**Online Learning** studies methods that make sequential decisions and update their parameters as new observations become available. Unlike one-off training on a fixed dataset, the protocol is sequential: **predict or decide → observe → incur a loss → update the model**.

The course introduces **Follow the Leader (FTL)**, **Follow the Regularized Leader (FTRL)**, and **Online Gradient Descent (OGD)**. Topics include convex losses, regularization (including Ridge), cumulative and average regret, and their theoretical guarantees. Applications cover demand forecasting and decision-making under asymmetric costs (*newsvendor*).

For a sequence of losses $\ell_1,\ldots,\ell_T$ and a decision set $\mathcal K$, **static regret** is defined as

$$ R_T = \sum_{t=1}^{T}\ell_t(\theta_t)- \min_{\theta\in\mathcal K}\sum_{t=1}^{T}\ell_t(\theta). $$

The objective is to achieve average regret $R_T/T$ converging to zero relative to the best **fixed** decision in hindsight. This alone does not guarantee a small absolute forecasting error.

### Available materials

| Material | Description |
|:--|:--|
| [Cours_Online_Learning.pdf](Cours_Online_Learning.pdf) | Lecture slides: sequential learning, FTL, FTRL, OGD, regret, and applications. |
| [Poly_Online_Learning.pdf](Poly_Online_Learning.pdf) | Lecture notes by Paul Minchella: definitions, mathematical developments, and examples. |
| [Illustration_FTL_FTRL_OGD.ipynb](Illustration_FTL_FTRL_OGD.ipynb) | Self-contained FTL, FTRL, and OGD example on synthetic demand, with squared loss and average regret. |
| [TP1_Bike/README.md](TP1_Bike/README.md) | Detailed description of Practical 1, its protocol, models, and data. |
| [TP1_Bike/TP1_Bike_template_ENG.ipynb](TP1_Bike/TP1_Bike_template_ENG.ipynb) | **Practical 1 — English student notebook**, to be completed (code and written answers). |
| [TP1_Bike/TP1_Bike_template_FR.ipynb](TP1_Bike/TP1_Bike_template_FR.ipynb) | **Practical 1 — French student notebook**, to be completed. |
| [TP1_Bike/requirements.txt](TP1_Bike/requirements.txt) | Python requirements for Practical 1. |
| [TP1_Bike/online_learning_bike_sharing_data/](TP1_Bike/online_learning_bike_sharing_data/) | UCI Bike Sharing dataset and provenance documentation. |

### Practical 1 — Bike Sharing: online forecasting and decisions

Practical 1 uses **daily bicycle rental counts in Washington, D.C.**, from the [UCI Bike Sharing dataset](https://doi.org/10.24432/C5W894), to investigate two complementary problems:

1. **Forecast demand** under squared loss using projected OGD and OGD with Ridge regularization.
2. **Choose an available quantity** under shortage and excess costs using OGD for the *newsvendor* loss, first without and then with covariates.

Students construct features known before each decision (calendar variables and lagged demand), check assumptions behind regret guarantees, implement the update rules, and compare them with naïve baselines. The evaluation uses a **chronological** split: online learning in 2011 and evaluation in 2012, always predicting **before** the day's demand is revealed.

The practical distinguishes forecasting metrics (**RMSE, MAE, bias**), operational *newsvendor* cost, and **average regret**. Optimizing squared error and minimizing asymmetric costs need not lead to the same decision: the latter targets a critical quantile rather than a mean.

**For students:** both notebooks are **unfinished templates**, not directly executable solutions. Preserve the required function names and signatures, and fill in the written-answer cells. Solution notebooks are not distributed in this folder.

### Setup and execution

**Requirements:** Python **3.10+**, a Jupyter environment (JupyterLab or VS Code), and the practical's dependencies. From a **working directory** where you want to clone the repository:

```bash
git clone https://github.com/MINCHELLA-Paul/Master-MIASHS.git
cd Master-MIASHS/M2_OnlineLearning
python -m pip install -r TP1_Bike/requirements.txt
python -m pip install jupyterlab
```

Open the FTL/FTRL/OGD illustration with:

```bash
python -m jupyterlab Illustration_FTL_FTRL_OGD.ipynb
```

For Practical 1, start Jupyter **in the practical's directory** so relative data paths resolve correctly:

```bash
cd TP1_Bike
python -m jupyterlab TP1_Bike_template_ENG.ipynb
# Or open the French edition:
# python -m jupyterlab TP1_Bike_template_FR.ipynb
```

The daily dataset is provided in the practical's data directory; the notebook can also retrieve it from UCI if the local CSV is missing. See the [Practical 1 README](TP1_Bike/README.md) for details and instructions.

**Suggested learning path:** lecture slides and notes → FTL/FTRL/OGD illustration → Bike Sharing practical (English or French).
