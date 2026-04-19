# Examen d'entraînement -- Statistique & Machine Learning
# Chargement des packages ####
library(FactoMineR)
library(corrplot)
library(MASS)

# Import des données ####
setwd("~/Desktop/TRAVAIL/EPITA/2025_03 EPITA4 PBS2/STATISTIQUE/Blank Exam")
df <- read.csv(
  "UCI_wine.txt",
  sep = ",",
  header = TRUE
)

# Partie I -- Statistiques descriptives ####
## Question 1 ####
# Dimensions du dataset
dim(df)
nrow(df)
ncol(df)



## Question 2 ####
# Nature des variables
str(df)
vars_quali <- df[, "Class"]
vars_quanti <- df[, -1]


## Question 3 ####
# Moyenne et écart-type de Alcohol
mean(df$Alcohol)
sd(df$Alcohol)


## Question 4 ####
# Corrélation Alcohol / Proline
cor(df$Alcohol, df$Proline)

## Question 5 ####
# Valeurs manquantes et valeurs atypiques
sum(is.na(df))
summary(df)

par(mar = c(10, 4, 4, 2))

boxplot(
  scale(vars_quanti),
  las = 2,
  col = "#3DCEB7",
  main = "Boxplots des variables quantitatives standardisées"
)

par(mar = c(5, 4, 4, 2))

## Question 6 ####
# Individus au-dessus de la médiane de Y
df[df$Alcohol > median(df$Alcohol), ]
which(df$Alcohol > median(df$Alcohol))


## Question 7 ####
# Matrice de corrélation
corrplot(
  cor(vars_quanti),
  method = "circle",
  type = "upper",
  tl.col = "black",
  tl.cex = 0.7
)


## Question 8 ####
# Couple de variables avec corrélation maximale
cor_mat <- cor(vars_quanti)
cor_mat[upper.tri(cor_mat, diag = TRUE)] <- NA
idx <- which(abs(cor_mat) == max(abs(cor_mat), na.rm = TRUE), arr.ind = TRUE)
c(rownames(cor_mat)[idx[1]], colnames(cor_mat)[idx[2]])

# Partie II -- Régression linéaire ####
## Question 9 ####
# Modèle linéaire
model <- lm(Alcohol ~ ., data = vars_quanti)



## Question 10 ####
# Résumé du modèle
summary(model)


## Question 11 ####
# Variable la moins significative
summary(model)$coefficients


## Question 12 ####
# R2 et R2 ajusté
summary(model)$r.squared
summary(model)$adj.r.squared


## Question 13 ####
# Écart-type des résidus
summary(model)$sigma


## Question 14 ####
# Prédiction nouvelle observation
# Construction de la nouvelle observation ####
new_obs <- data.frame(
  Malic = 2.34,
  Ash = 2.37,
  Alcalinity_of_Ash = 19.49,
  Magnesium = 99.74,
  Total_Phenols = 2.30,
  Flavanoids = 2.03,
  Nonflavanoid_Phenols = 0.36,
  Proanthocyanins = 1.59,
  Color_Intensity = 5.06,
  Hue = 0.96,
  OD280_OD315 = 2.61,
  Proline = 746.89
)


# Prédiction ####
Y_new <- predict(model, newdata = new_obs)

# Arrondi à deux décimales ####
round(Y_new, 2)




# Partie III -- ACP ####
## Question 15 ####
# ACP sur variables explicatives uniquement
X <- df[, -which(names(df) == "Alcohol")]
Wine.PCA <- PCA(
  X,
  scale.unit = TRUE,
  quali.sup = 1,
  ncp = 12,
  graph = TRUE
)



## Question 16 ####
# Variance expliquée axes 1 et 2
Wine.PCA$eig[1:2, ]


## Question 17 ####
# Variables contribuant le plus aux axes 1 et 2
sort(Wine.PCA$var$contrib[, 1], decreasing = TRUE)[1:5]
sort(Wine.PCA$var$contrib[, 2], decreasing = TRUE)[1:5]



## Question 18 ####
# Individus contribuant le plus aux axes 1 et 2
sort(Wine.PCA$ind$contrib[, 1], decreasing = TRUE)[1:5]
sort(Wine.PCA$ind$contrib[, 2], decreasing = TRUE)[1:5]



## Question 19 ####
# Coordonnées variables pour interprétation axes
Wine.PCA$var$coord[, 1:2]


# Partie IV -- Réduction de dimension ####
## Question 20 ####
# Variance expliquée cumulée par 3 composantes
Wine.PCA$eig[3, 3]



## Question 21 ####
# Régression sur composantes principales
PC <- Wine.PCA$ind$coord[, 1:3]

df_PCA <- data.frame(
  Alcohol = df$Alcohol,
  PC
)

model_PCA <- lm(
  Alcohol ~ .,
  data = df_PCA
)



## Question 22 ####
# Nouveau R2 et R2 ajusté
summary(model_PCA)$r.squared
summary(model_PCA)$adj.r.squared



## Question 23 ####
# Comparaison des modèles
summary(model)
summary(model_PCA)


# Partie V -- Analyse discriminante linéaire (LDA) ####
## Question 24 ####
# Table de contingence de la variable Class
table(df$Class)


## Question 25 ####
# Modèle LDA avec 3 variables
lda_model_3var <- lda(
  Class ~ Flavanoids + Color_Intensity + Proline,
  data = df
)

lda_model_3var



## Question 26 ####
# Matrice de confusion associée au modèle
pred_3var <- predict(lda_model_3var)
conf_mat_3var <- table(
  Observed = df$Class,
  Predicted = pred_3var$class
)
conf_mat_3var


# Nombre de bien classés
sum(diag(conf_mat_3var))


# Nombre de mal classés
sum(conf_mat_3var) - sum(diag(conf_mat_3var))



## Question 27 ####
# Observations mal classées (1 ligne demandée)
which(pred_3var$class != df$Class)



## Question 28 ####
# Observation mal classée dont la probabilité a posteriori
# d'appartenir à sa vraie classe est minimale
wrong_idx <- which(pred_3var$class != df$Class)
true_class_prob <- pred_3var$posterior[
  cbind(wrong_idx, df$Class[wrong_idx])
]
worst_obs <- wrong_idx[which.min(true_class_prob)]
worst_obs

# Valeur correspondante
min(true_class_prob)


## Question 29 ####
# Nouvelle observation
new_wine <- data.frame(
  Flavanoids = 2.50,
  Color_Intensity = 5.20,
  Proline = 900
)

pred_new <- predict(
  lda_model_3var,
  newdata = new_wine
)
pred_new$class


# Probabilités a posteriori associées
pred_new$posterior



## Question 30 ####
# Modèle LDA avec toutes les variables explicatives
lda_model_full <- lda(
  Class ~ .,
  data = df
)
lda_model_full



## Question 31 ####
# Matrice de confusion modèle complet
pred_full <- predict(lda_model_full)
conf_mat_full <- table(
  Observed = df$Class,
  Predicted = pred_full$class
)
conf_mat_full


# Nombre de bien classés
sum(diag(conf_mat_full))


# Nombre de mal classés
sum(conf_mat_full) - sum(diag(conf_mat_full))
