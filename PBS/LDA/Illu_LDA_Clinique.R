# Packages nécessaires
library(MASS)
library(ggplot2)

set.seed(1)

# Générer les données ####

# Nombre d'observations par classe
n <- 120

# Centres des deux classes
mu_benign  <- c(18, 0.25)
mu_malign  <- c(42, 0.65)

# Covariance commune (hypothèse LDA)
Sigma <- matrix(
  c(60, 1.2,
    1.2, 0.2),
  nrow = 2
)

# Simulation des données
Z_benign  <- mvrnorm(n, mu_benign, Sigma)
Z_malign  <- mvrnorm(n, mu_malign, Sigma)

# Dataframe final
df <- rbind(
  data.frame(Z1 = Z_benign[,1],
             Z2 = Z_benign[,2],
             Classe = "Tumeur bénigne"),
  
  data.frame(Z1 = Z_malign[,1],
             Z2 = Z_malign[,2],
             Classe = "Cancer malin")
)


 
## Plot 1 : nuage seul ####

ggplot(df, aes(x = Z1, y = Z2, color = Classe)) +
  
  geom_point(size = 2.5, alpha = 0.8) +
  
  labs(
    title = "Deux populations simulées : tumeur bénigne vs cancer malin",
    x = "Taille tumorale (mm)",
    y = "Indice d’irrégularité du contour"
  ) +
  
  theme_minimal(base_size = 14) +
  
  scale_color_manual(values = c(
    "Tumeur bénigne" = "#2E86C1",
    "Cancer malin"   = "#C0392B"
  ))

# Calcul des barycentres

centres <- aggregate(cbind(Z1, Z2) ~ Classe, df, mean)


## Plot 2 : nuage + ellipses + barycentres ####
ggplot(df, aes(x = Z1, y = Z2, color = Classe)) +
  geom_point(size = 2.5, alpha = 0.8) +
  stat_ellipse(level = 0.95, linewidth = 1.2) +
  geom_point(
    data = centres,
    aes(x = Z1, y = Z2),
    size = 5,
    shape = 4,
    stroke = 2
  ) +
  labs(
    title = "Illustration du modèle LDA : ellipses de dispersion et centres de classes",
    x = "Taille tumorale (mm)",
    y = "Indice d’irrégularité du contour"
  ) +
  theme_minimal(base_size = 14) +
  scale_color_manual(values = c(
    "Tumeur bénigne" = "#2E86C1",
    "Cancer malin"   = "#C0392B"
  ))

# LDA ####
# Apprentissage du modèle LDA
lda_model <- lda(Classe ~ Z1 + Z2, data = df)
lda_model


# Prédiction sur la base d'apprentissage ####
# Prédictions
lda_pred <- predict(lda_model)
lda_pred

# Probabilités a posteriori
head(lda_pred$posterior)

# Performances sur la base d'apprentissage ####
# Matrice de confusion
conf_mat <- table(
  Classe_reelle = df$Classe,
  Classe_predite = lda_pred$class
)
conf_mat

# Accuracy
accuracy <- mean(lda_pred$class == df$Classe)
accuracy


# Performances sur un dataset de test indépendant ####
## Jeu de test indépendant ####
n_test <- 25
Z_benign_test <- mvrnorm(n_test, mu_benign, Sigma)
Z_malign_test <- mvrnorm(n_test, mu_malign-c(5,0.3), Sigma)

df_test <- rbind(
  data.frame(
    Z1 = Z_benign_test[, 1],
    Z2 = Z_benign_test[, 2],
    Classe = "Tumeur bénigne"
  ),
  data.frame(
    Z1 = Z_malign_test[, 1],
    Z2 = Z_malign_test[, 2],
    Classe = "Cancer malin"
  )
)

df$Classe <- factor(df$Classe,
                    levels = c("Tumeur bénigne", "Cancer malin"))

df_test$Classe <- factor(df_test$Classe,
                         levels = c("Tumeur bénigne", "Cancer malin"))



## Prédiction sur le jeu de test ####
lda_pred_test <- predict(lda_model, newdata = df_test)

## Matrice de confusion Test ####
conf_mat_test <- table(
  Classe_reelle = df_test$Classe,
  Classe_predite = lda_pred_test$class
)
conf_mat_test


## Accuracy Test ####
accuracy_test <- mean(lda_pred_test$class == df_test$Classe)
accuracy_test

## Plot 3 : apprentissage + test + frontière bayésienne ####
# Barycentres du jeu d'apprentissage
centres <- aggregate(cbind(Z1, Z2) ~ Classe, df, mean)

# Grille pour tracer la frontière de décision bayésienne
grid_df <- expand.grid(
  Z1 = seq(min(c(df$Z1, df_test$Z1)) - 5, max(c(df$Z1, df_test$Z1)) + 5, length.out = 400),
  Z2 = seq(min(c(df$Z2, df_test$Z2)) - 0.5, max(c(df$Z2, df_test$Z2)) + 0.5, length.out = 400)
)

grid_post <- predict(lda_model, newdata = grid_df)$posterior
grid_df$posterior_malign <- grid_post[, "Cancer malin"]

ggplot() +
  
  # Jeu d'apprentissage
  geom_point(
    data = df,
    aes(x = Z1, y = Z2, color = Classe),
    size = 2.5,
    alpha = 0.8
  ) +
  
  # Ellipses du jeu d'apprentissage
  stat_ellipse(
    data = df,
    aes(x = Z1, y = Z2, color = Classe),
    level = 0.95,
    linewidth = 1.2
  ) +
  
  # Barycentres du jeu d'apprentissage
  geom_point(
    data = centres,
    aes(x = Z1, y = Z2, color = Classe),
    size = 5,
    shape = 4,
    stroke = 2
  ) +
  
  # Jeu de test bénin
  geom_point(
    data = subset(df_test, Classe == "Tumeur bénigne"),
    aes(x = Z1, y = Z2),
    color = "#3DCEB7",
    size = 3,
    shape = 17
  ) +
  
  # Jeu de test malin
  geom_point(
    data = subset(df_test, Classe == "Cancer malin"),
    aes(x = Z1, y = Z2),
    color = "purple",
    size = 3,
    shape = 17
  ) +
  
  # Frontière de décision bayésienne : P(Cancer malin | z) = 0.5
  geom_contour(
    data = grid_df,
    aes(x = Z1, y = Z2, z = posterior_malign),
    breaks = 0.5,
    color = "black",
    linewidth = 1.2,
    linetype = "dashed"
  ) +
  
  labs(
    title = "LDA : apprentissage, jeu de test et frontière de décision bayésienne",
    subtitle = "Train : couleurs par classe | Test bénin : teal | Test malin : purple",
    x = "Taille tumorale (mm)",
    y = "Indice d’irrégularité du contour",
    color = "Jeu d'apprentissage"
  ) +
  
  theme_minimal(base_size = 14) +
  
  scale_color_manual(values = c(
    "Tumeur bénigne" = "#2E86C1",
    "Cancer malin"   = "#C0392B"
  ))