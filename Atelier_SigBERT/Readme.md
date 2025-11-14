# Atelier SigBERT

Ce sous-dépôt constitue un **atelier pratique** destiné aux étudiants de Master 2 MIASHS dans le cadre du cours d’**Atelier Data Science (Analyse de survie)**.

---

## 1. Objectif général

L’objectif de cet atelier est double :

1. **Mettre en œuvre un modèle de survie pénalisé** à partir de données prétraitées issues du projet **SigBERT**, déjà transformées sous forme de coefficients de signatures.
2. **Explorer la prédiction conforme en analyse de survie**, en particulier pour :
   - produire des **intervalles prédictifs** ou des **bandes de confiance conformes** sur le **score de risque** $$\hat{\eta} = \beta \cdot \mathbb{S},$$
   - ou sur la **probabilité de survie à un temps donné** $\mathbb{P}(T > t^\star)$,
   - ou tout autre score de votre choix,
   en discutant quelle cible est la plus pertinente dans un cadre clinique et statistique.

L’approche conforme doit ici être **conceptuellement réfléchie et mise en œuvre par les étudiants** : il s’agit de comprendre ce que signifie une garantie de couverture en survie et comment elle peut être interprétée sur des données médicales.

---

## 2. Données disponibles

Les deux fichiers de travail proposés sont : `df_study_L18_w6.csv` stocké dans `df_study_selected.zip` et `df_study_L36_w6.csv` stocké dans `df_study_selected_L36_w6.zip`.  
**Attention** : il faudra retrancher à la variable `time` $18\times 30$ (days) pour le dataset issu de `df_study_selected.zip`; sinon $36\times 30$ (days)

- Il faut établir les statistiques descriptives. (n_obs; Mean, Std; min, Q1, Med, Q3, Max; histogramme des variables pertinentes, etc.) 
- Chaque ligne correspond à un patient (ou à une unité d’analyse temporelle agrégée).
- Les colonnes incluent :
  - un identifiant anonymisé `ID`,
  - les coefficients de signatures extraits via SigBERT,
  - les variables de survie : `event` (indicateur de décès : True = Décédé, False = censoré) et `time` (durée de suivi en jours).

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
- estimation du risque via un **modèle de Cox régularisé**.

Le dépôt GitHub correspondant est accessible ici :  
[https://github.com/MINCHELLA-Paul/SigBERT](https://github.com/MINCHELLA-Paul/SigBERT)

---

## 5. Travail attendu

1. Charger un des deux jeux de données `df_study_selected.zip` ou `df_study_selected_L36_w8.zip`.
2. Ajuster plusieurs modèles de survie et évaluer leurs performances.
3. Concevoir une procédure de **prédiction conforme** :
   - sur le **score de risque individuel** \(\hat{\eta}\),
   - ou sur la **probabilité de survie conditionnelle** à un temps \(t^\star\),
   - ou le score de votre choix, justifié.
4. Discuter :
   - quelle forme de prédiction conforme semble la plus cohérente,
   - comment interpréter la couverture obtenue dans un cadre médical,
   - quelles limites méthodologiques peuvent survenir (censure, dépendances, etc.).

---

## 6. Structure minimale du répertoire

```
Atelier_SigBERT/
│
├── df_study_selected.csv # Données de l’étude (anonymisées)
├── README.md # Présent document
└── notebooks/
└── votre_notebook_ici.ipynb # Exemple d'analyse de survie avec garantie conforme
```
