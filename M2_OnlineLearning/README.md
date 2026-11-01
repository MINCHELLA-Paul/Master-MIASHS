# M2 — Online Learning / Apprentissage en ligne

<p align="center">
  <img src="Logo_Oventi_ERIC_Lyon2.png" alt="Logos Oventi, ERIC et Université Lumière Lyon 2" width="600">
</p>

---

## Français

Supports du cours d’**apprentissage en ligne** du Master 2 MIASHS, Université Lumière Lyon 2.

- **Principe :** prédire, observer le résultat, puis mettre à jour le modèle à chaque nouvelle donnée.
- **Méthodes :** Follow the Leader (FTL), Follow the Regularized Leader (FTRL) et Online Gradient Descent (OGD) ; perte quadratique, régularisation et regret.
- **Applications :** illustration sur une demande simulée et étude UCI Bike Sharing ; évaluation chronologique, prévention des fuites de données et distinction entre regret et erreur de prévision.

**Parcours conseillé :** diapositives → notebook d’illustration → notebook étudiant Bike Sharing (FR ou EN), accompagné du guide des exercices.

### Contenu du dépôt

| Support | Description |
| --- | --- |
| [Cours_Online_Learning.pdf](Cours_Online_Learning.pdf) | Diapositives du cours |
| [Poly_Online_Learning.pdf](Poly_Online_Learning.pdf) | Polycopié de cours de Paul Minchella |
| [Illustration_FTL_FTRL_OGD.ipynb](Illustration_FTL_FTRL_OGD.ipynb) | Illustration autonome des méthodes FTL, FTRL et OGD |
| [TP1_Bike/Student_Template_EN.ipynb](TP1_Bike/Student_Template_EN.ipynb) | Notebook étudiant en anglais à compléter |
| [TP1_Bike/Student_Template_FR.ipynb](TP1_Bike/Student_Template_FR.ipynb) | Notebook étudiant en français à compléter |
| [TP1_Bike/Guide_Exercices_1_12_FR.pdf](TP1_Bike/Guide_Exercices_1_12_FR.pdf) | Guide en français des exercices 1 à 12 |
| [TP1_Bike/README.md](TP1_Bike/README.md) | Documentation détaillée du TP |
| [TP1_Bike/requirements.txt](TP1_Bike/requirements.txt) | Dépendances Python du TP |
| [TP1_Bike/online_learning_bike_sharing_data/](TP1_Bike/online_learning_bike_sharing_data/) | Données UCI, archive et provenance |

**Note :** le README du TP et le guide font référence à un notebook de corrigé qui n’est pas inclus dans ce dépôt.

### Démarrage

Python 3.10 ou version ultérieure, depuis la racine du dépôt :

```bash
python -m pip install -r TP1_Bike/requirements.txt
python -m pip install jupyterlab
python -m jupyterlab Illustration_FTL_FTRL_OGD.ipynb
```

---

## English

Materials for the **Online Learning** course in the MIASHS Master’s programme (M2), Université Lumière Lyon 2.

- **Principle:** predict, observe the outcome, then update the model as each new observation arrives.
- **Methods:** Follow the Leader (FTL), Follow the Regularized Leader (FTRL), and Online Gradient Descent (OGD); squared loss, regularization, and regret.
- **Applications:** a simulated-demand illustration and the UCI Bike Sharing case study; chronological evaluation, data leakage prevention, and the distinction between regret and prediction error.

**Suggested order:** lecture slides → illustration notebook → Bike Sharing student notebook (EN or FR), alongside the exercise guide.

### Repository contents

| Material | Description |
| --- | --- |
| [Cours_Online_Learning.pdf](Cours_Online_Learning.pdf) | Lecture slides |
| [Poly_Online_Learning.pdf](Poly_Online_Learning.pdf) | Lecture notes by Paul Minchella |
| [Illustration_FTL_FTRL_OGD.ipynb](Illustration_FTL_FTRL_OGD.ipynb) | Self-contained illustration of FTL, FTRL, and OGD |
| [TP1_Bike/Student_Template_EN.ipynb](TP1_Bike/Student_Template_EN.ipynb) | English student notebook to complete |
| [TP1_Bike/Student_Template_FR.ipynb](TP1_Bike/Student_Template_FR.ipynb) | French student notebook to complete |
| [TP1_Bike/Guide_Exercices_1_12_FR.pdf](TP1_Bike/Guide_Exercices_1_12_FR.pdf) | French guide covering exercises 1–12 |
| [TP1_Bike/README.md](TP1_Bike/README.md) | Detailed practical documentation |
| [TP1_Bike/requirements.txt](TP1_Bike/requirements.txt) | Python dependencies for the practical |
| [TP1_Bike/online_learning_bike_sharing_data/](TP1_Bike/online_learning_bike_sharing_data/) | UCI dataset, archive, and provenance |

**Note:** the practical README and exercise guide refer to a solution notebook that is not included in this repository.

### Getting started

Python 3.10 or later, from the repository root:

```bash
python -m pip install -r TP1_Bike/requirements.txt
python -m pip install jupyterlab
python -m jupyterlab Illustration_FTL_FTRL_OGD.ipynb
```
