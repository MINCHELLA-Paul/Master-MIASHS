# Atelier_SigBERT

Ce sous-dépôt constitue un **atelier pratique** destiné aux étudiants de Master 2 MIASHS dans le cadre du cours d’**Analyse de survie pénalisée**.

---

## 1. Objectif général

L’objectif de cet atelier est double :

1. **Mettre en œuvre un modèle de survie pénalisé (Cox-LASSO)** à partir de données prétraitées issues du projet **SigBERT**, déjà transformées sous forme de coefficients de signatures dans le fichier `df_study_all.csv`.
2. **Explorer la prédiction conforme en analyse de survie**, en particulier pour :
   - produire des **intervalles prédictifs** ou des **bandes de confiance conformes** sur le **score de risque**
     $$
     \hat{\eta} = \beta \cdot \mathbb{S},
     $$
   - ou sur la **probabilité de survie à un temps donné** \(\mathbb{P}(T > t^\star)\),
   - ou encore sur une **métrique de qualité du modèle** (comme le C-index ou le td-AUC),
   en discutant quelle cible est la plus pertinente dans un cadre clinique et statistique.

L’approche conforme doit ici être **conceptuellement réfléchie et mise en œuvre par les étudiants** : il s’agit de comprendre ce que signifie une garantie de couverture en survie et comment elle peut être interprétée sur des données médicales.

---

## 2. Données disponibles

Le fichier de travail principal est : `df_study_all.csv`.


- Il contient environ 7 000 observations et 185 colonnes.
- Chaque ligne correspond à un patient (ou à une unité d’analyse temporelle agrégée).
- Les colonnes incluent :
  - un identifiant anonymisé `ID`,
  - les coefficients de signatures extraits via SigBERT,
  - les variables de survie : `event` (indicateur de décès) et `time` (durée de suivi).

Ces données sont prêtes à être utilisées directement dans un modèle de Cox, ou dans toute autre approche de survie compatible avec un format tabulaire.

---

## 3. Lien avec le cours d’Analyse de survie

Le support de cours et les exemples d’implémentation de modèles de survie pénalisés sont disponibles sur le dépôt suivant :  
[https://github.com/MINCHELLA-Paul/Master-MIASHS/tree/main/Analyse_Survie_M2](https://github.com/MINCHELLA-Paul/Master-MIASHS/tree/main/Analyse_Survie_M2)

Ce cours fournit le socle méthodologique : modèles de Cox, régularisation LASSO, validation croisée, et métriques de performance.

---

## 4. Lien avec le projet SigBERT

Les données utilisées ici sont dérivées du projet **SigBERT**, une approche de modélisation en survie combinant :

- embeddings de texte clinique extraits avec **OncoBERT**,  
- compression dimensionnelle (PCA ou Johnson–Lindenstrauss),  
- extraction de **signatures de chemins** pour modéliser la dynamique temporelle,  
- estimation du risque via un **modèle de Cox régularisé (LASSO)**.

Le dépôt GitHub correspondant est accessible ici :  
[https://github.com/MINCHELLA-Paul/SigBERT](https://github.com/MINCHELLA-Paul/SigBERT)

---

## 5. Travail attendu

1. Charger le jeu de données `df_study_all.csv`.
2. Ajuster un modèle de **Cox-LASSO** et évaluer ses performances.
3. Concevoir une procédure de **prédiction conforme** :
   - sur le **score de risque individuel** \(\hat{\eta}\),
   - ou sur la **probabilité de survie conditionnelle** à un temps \(t^\star\),
   - ou sur une **métrique d’évaluation** (ex. c-index).
4. Discuter :
   - quelle forme de prédiction conforme semble la plus cohérente,
   - comment interpréter la couverture obtenue dans un cadre médical,
   - quelles limites méthodologiques peuvent survenir (censure, dépendances, etc.).

---

## 6. Structure minimale du répertoire

```
Atelier_SigBERT/
│
├── df_study_all.csv # Données de l’étude (anonymisées)
├── README.md # Présent document
└── notebooks/
├── Cox_LASSO.ipynb # Exemple d'analyse de survie pénalisée
└── Conformal_Prediction.ipynb # Atelier sur la prédiction conforme
```


---

## 7. Conseils méthodologiques

- Penser à **standardiser les covariables** avant la régression pénalisée.
- Utiliser **validation croisée** pour le choix du paramètre de régularisation.
- Pour la prédiction conforme :
  - lire les articles récents sur *Conformalized Survival Analysis (CSA)* et *Conformalized Survival Distributions (CSD)*,
  - réfléchir à la variable de sortie sur laquelle appliquer la couverture.

---

## 8. Licence et attribution

Ce matériel pédagogique est fourni à titre académique pour les étudiants du Master MIASHS.  
Les données sont **anonymisées** et **issues du projet SigBERT**, développé à des fins de recherche en modélisation du risque patient.
