# TP LDA

# Question 1 - Chargement du package ####
library(MASS)

# Question 2 - Chargement des données ####
data(iris)
df <- iris; X <- df[, 1:4]; y <- df$Species

# Aperçu rapide
head(df); str(df); summary(df)

## Pairplot des variables quantitatives ####
pairs(
  iris[, 1:4],
  col = as.numeric(iris$Species),
  pch = 19,
  main = "Pairplot des variables du dataset iris"
)

legend(
  "topright",
  legend = levels(iris$Species),
  col = 1:3,
  pch = 19,
  bty = "n"
)


## Histogrammes séparés par classe (mfrow 2x2) ####
par(mfrow = c(2, 2))

vars <- names(iris)[1:4]
cols <- c("red", "green3", "blue")

for (v in vars) {
  hist(
    iris[iris$Species == "setosa", v],
    col = adjustcolor(cols[1], alpha.f = 0.5),
    main = paste("Histogramme de", v),
    xlab = v,
    xlim = range(iris[, v]),
    breaks = 10
  )
  hist(
    iris[iris$Species == "versicolor", v],
    col = adjustcolor(cols[2], alpha.f = 0.5),
    add = TRUE,
    breaks = 10
  )
  hist(
    iris[iris$Species == "virginica", v],
    col = adjustcolor(cols[3], alpha.f = 0.5),
    add = TRUE,
    breaks = 10
  )
  legend(
    "topright",
    legend = levels(iris$Species),
    fill = adjustcolor(cols, alpha.f = 0.5),
    bty = "n"
  )
}
par(mfrow = c(1, 1))


# Question 3 - Instance et Apprentissage LDA ####
lda_model <- lda(Species ~ ., data = df)

# Résumé du modèle ajusté
lda_model


# Question 4 - Explorer le modèle lda appris ####
# Proportions a priori estimées par R
lda_model$prior

# Décompte de chaque espèce
table(df$Species)

# Nombre total d'observations
nrow(df)

# Centres de gravité de chaque classe
lda_model$means

# (Hors programme visiblement) 
# Coefficients discriminants utilisés par R
lda_model$scaling

# (Hors programme visiblement)
# Valeurs singulières / importance des axes discriminants
lda_model$svd

# Commentaire possible :
# - lda_model$prior donne les proportions des classes
# - table(df$Species) donne les effectifs
# - lda_model$means donne les centres de classes
# - lda_model$scaling donne les coefficients des fonctions discriminantes

# Question 5 - Predict de la LDA ####
lda_pred <- predict(lda_model, newdata = X)

# Structure de l'objet retourné
names(lda_pred)

# Classes prédites
head(lda_pred$class)

# Scores sur les axes discriminants
head(lda_pred$x)

# Probabilités a posteriori
head(lda_pred$posterior)

# Question 6 - Matrice de confusion ####
conf_mat_lda <- table(
  Classe_reelle = y,
  Classe_predite = lda_pred$class
)
conf_mat_lda

# Taux de bonne classification
mean(lda_pred$class == y)

# Taux d'erreur
mean(lda_pred$class != y)


# Question 7 - Probabilités a posteriori estimées par la LDA ####
# Probabilités a posteriori calculées par R
posterior_probs <- lda_pred$posterior
head(posterior_probs)
tail(posterior_probs)

# Question 8 - Estimation sur le jeu d'apprentissage ####
# Vérification : la classe prédite est celle de probabilité a posteriori maximale
class_from_posterior <- colnames(posterior_probs)[max.col(posterior_probs)]

# Comparaison avec les prédictions de predict(...)
all(class_from_posterior == as.character(lda_pred$class))

# On peut aussi visualiser les premières lignes
verification_df <- data.frame(
  Classe_predite = lda_pred$class,
  Classe_via_posterior = class_from_posterior
)
head(verification_df)

# Question 9 - Inférence ####

# Quelques individus arbitraires
new_flowers <- data.frame(
  Sepal.Length = c(5.0, 6.2, 6.9, 5.8),
  Sepal.Width  = c(3.4, 2.8, 3.1, 2.7),
  Petal.Length = c(1.5, 4.8, 5.4, 5.1),
  Petal.Width  = c(0.2, 1.8, 2.1, 1.9)
)

new_pred <- predict(lda_model, newdata = new_flowers)

new_flowers
new_pred$class
new_pred$posterior
new_pred$x

# Résumé lisible
data.frame(
  new_flowers,
  Classe_predite = new_pred$class
)

# Question 10 - ACP et LDA ####

# ACP sur les variables quantitatives centrées-réduites
pca_iris <- prcomp(X, center = TRUE, scale. = TRUE)

summary(pca_iris)

# Coordonnées sur les deux premiers axes principaux
pca_scores <- pca_iris$x[, 1:2]

# Classification hiérarchique sur les coordonnées ACP
dist_mat <- dist(pca_scores)
hc <- hclust(dist_mat, method = "ward.D2")

# Découpage en 3 groupes
hc_groups <- cutree(hc, k = 3)

# Matrice de confusion brute
conf_mat_hc_raw <- table(
  Classe_reelle = y,
  Groupe_HC = hc_groups
)

conf_mat_hc_raw

# Comme les labels 1, 2, 3 des groupes HC sont arbitraires,
# on les renomme par majorité pour comparer proprement aux espèces.
map_group_to_species <- function(true_labels, cluster_labels) {
  tab <- table(cluster_labels, true_labels)
  mapping <- apply(tab, 1, function(v) names(which.max(v)))
  predicted_species <- mapping[as.character(cluster_labels)]
  factor(predicted_species, levels = levels(true_labels))
}

hc_species <- map_group_to_species(y, hc_groups)

conf_mat_hc <- table(
  Classe_reelle = y,
  Classe_predite = hc_species
)

conf_mat_hc

# Taux de bonne classification après renommage des groupes
mean(hc_species == y)

# Question 10 bis ####

# Représentation graphique de l'ACP
plot(
  pca_scores,
  col = as.numeric(y),
  pch = 19,
  xlab = "Axe principal 1",
  ylab = "Axe principal 2",
  main = "ACP des iris"
)
legend(
  "topright",
  legend = levels(y),
  col = 1:length(levels(y)),
  pch = 19,
  bty = "n"
)

# Dendrogramme de la classification hiérarchique
plot(
  hc,
  labels = FALSE,
  main = "Classification hiérarchique sur les scores ACP"
)
rect.hclust(hc, k = 3, border = 2:4)
