# M2 — Online Learning / Apprentissage en ligne

## Français

Supports du cours d’**apprentissage en ligne** du Master 2 MIASHS, Université Lumière Lyon 2.

- **Principe :** prédire, observer le résultat, puis mettre à jour le modèle à chaque nouvelle donnée.
- **Méthodes :** Follow the Leader (FTL), Follow the Regularized Leader (FTRL) et Online Gradient Descent (OGD) ; perte quadratique, régularisation et regret.
- **Applications :** illustration sur une demande simulée et étude UCI Bike Sharing ; évaluation chronologique, prévention des fuites de données et distinction entre regret et erreur de prévision.

Parcours conseillé : diapositives → notebook d’illustration → notebook étudiant Bike Sharing (FR ou EN), accompagné du guide des exercices.

## English

Materials for the **Online Learning** course in the MIASHS Master’s programme (M2), Université Lumière Lyon 2.

- **Principle:** predict, observe the outcome, then update the model as each new observation arrives.
- **Methods:** Follow the Leader (FTL), Follow the Regularized Leader (FTRL), and Online Gradient Descent (OGD); squared loss, regularization, and regret.
- **Applications:** a simulated-demand illustration and the UCI Bike Sharing case study; chronological evaluation, data leakage prevention, and the distinction between regret and prediction error.

Suggested order: slides → illustration notebook → Bike Sharing student notebook (EN or FR), alongside the exercise guide.

## Fichiers / Files

| Support / Material | Description FR / EN |
| --- | --- |
| [Cours_Online_Learning.pdf](Cours_Online_Learning.pdf) | Diapositives du cours / Lecture slides |
| [Poly_Online_Learning.pdf](Poly_Online_Learning.pdf) | Polycopié de cours de Paul Minchella / Lecture notes by Paul Minchella |
| [Illustration_FTL_FTRL_OGD.ipynb](Illustration_FTL_FTRL_OGD.ipynb) | Illustration autonome de FTL, FTRL et OGD / Self-contained FTL, FTRL, and OGD illustration |
| [TP1_Bike/Student_Template_EN.ipynb](TP1_Bike/Student_Template_EN.ipynb) | Notebook étudiant en anglais à compléter / English student notebook to complete |
| [TP1_Bike/Student_Template_FR.ipynb](TP1_Bike/Student_Template_FR.ipynb) | Notebook étudiant en français à compléter / French student notebook to complete |
| [TP1_Bike/Guide_Exercices_1_12_FR.pdf](TP1_Bike/Guide_Exercices_1_12_FR.pdf) | Guide en français des exercices 1–12 / French guide to exercises 1–12 |
| [TP1_Bike/README.md](TP1_Bike/README.md) | Documentation détaillée du TP / Detailed practical documentation |
| [TP1_Bike/requirements.txt](TP1_Bike/requirements.txt) | Dépendances Python du TP / Practical Python dependencies |
| [TP1_Bike/online_learning_bike_sharing_data/](TP1_Bike/online_learning_bike_sharing_data/) | Données UCI, archive et provenance / UCI data, archive, and provenance |

**Note :** le README du TP et le guide font référence à un notebook de corrigé non inclus ici. / The practical README and guide refer to a solution notebook that is not included here.

## Démarrage / Getting started

Python 3.10+, depuis ce dossier / from this folder:

```bash
python -m pip install -r TP1_Bike/requirements.txt
python -m pip install jupyterlab
python -m jupyterlab Illustration_FTL_FTRL_OGD.ipynb
```
