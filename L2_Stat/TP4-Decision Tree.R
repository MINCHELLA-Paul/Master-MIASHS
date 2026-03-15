# EXERCICE 1 -- ARBRE DE DECISION ####

# Chargement des bibliothèques
library(rpart); library(rpart.plot); library(ggplot2)

# Chargement des données
data(mtcars)

df <- mtcars[, c("wt", "hp", "mpg")]
  
head(df)

par(mfrow = c(1,1))

# Apprentissage de l'arbre de régression
tree <- rpart(mpg ~ wt + hp, data = df)

# Affichage de l'arbre de décision
rpart.plot(tree)

# Création d'une grille de l'espace (wt, hp)
grid <- expand.grid(
  wt = seq(min(df$wt), max(df$wt), length.out = 200),
  hp = seq(min(df$hp), max(df$hp), length.out = 200)
)

# Prédictions du modèle sur la grille
grid$pred <- predict(tree, newdata = grid)

# Définition des régions (feuilles) induites par l'arbre
grid$region <- as.factor(round(grid$pred,2))

# Visualisation des régions et des données
ggplot() +
  geom_tile(
    data = grid,
    aes(x = wt, y = hp, fill = region),
    alpha = 0.4
  ) +
  geom_point(
    data = df,
    aes(x = wt, y = hp),
    size = 2
  ) +
  theme_minimal() +
  labs(
    x = "Poids (wt)",
    y = "Puissance (hp)",
    fill = "Régions"
  )


# Inférence pour une nouvelle observation
predict(tree, newdata = data.frame(wt = 1.7, hp = 63))


# EXERCICE 2 -- ARBRE DE DECISION ####
library(rpart)
library(rpart.plot)

# Création du dataframe à partir des données de l'exercice
data_ex2 <- data.frame(
  X1 = c(1, 0, 0, 0, 1),  # Large
  X2 = c(0, 0, 1, 0, 0),  # Hard
  X3 = c(0, 0, 1, 1, 1),  # Symmetric
  Y  = c(1, 1, 1, 0, 0)   # Maligne
)

# Apprentissage de l'arbre de décision
# (méthode "class" car Y est binaire)
tree <- rpart(
  Y ~ X1 + X2 + X3,
  data = data_ex2,
  method = "class",
  control = rpart.control(cp = 0, minsplit = 1)
)

# Affichage de l'arbre appris
rpart.plot(tree)


# EXERCICE 3 -- ARBRE DE DECISION ####
# Chargement des bibliothèques
library(rpart)
library(rpart.plot)


# Données de l'exercice 3
data_ex3 <- data.frame(
  X1 = factor(
    c("C","R","R","C","P","R","P","P","C","P","R","P","R"),
    levels = c("C","P","R")
  ),  # Type de musique
  X2 = factor(
    c(1,0,1,0,0,1,1,0,1,1,1,0,1),
    levels = c(0,1)
  ),  # Prix : 0 = bas, 1 = élevé
  Y = factor(
    c(1,1,1,1,1,0,0,0,1,0,0,1,0),
    levels = c(0,1)
  )   # Transaction
)


# Apprentissage de l'arbre

tree <- rpart(
  Y ~ X1 + X2,
  data = data_ex3,
  method = "class",
  parms = list(split = "information"),  # critère entropie
  control = rpart.control(cp = 0, minsplit = 1)
)


# Affichage de l'arbre

rpart.plot(
  tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE
)


# EXERCICE 4 -- ARBRE DE DECISION ####

library(rpart)
library(rpart.plot)

data <- data.frame(
  meteo=c("E","E","N","P","P","P","N","E","E","P","E","N","N","P"),
  temp=c("C","C","C","M","F","F","F","M","F","M","M","M","C","M"),
  humid=c(1,1,1,1,0,0,0,1,0,0,0,1,0,1),
  soldes=c(1,0,1,1,1,0,0,1,1,1,0,0,1,0),
  achats=c(25,30,46,45,52,23,43,35,38,46,48,52,44,30)
)

# Conversion en facteurs pour les variables qualitatives
data$meteo  <-  as.factor(data$meteo)
data$temp   <-  as.factor(data$temp)
data$humid  <-  as.factor(data$humid)
data$soldes <-  as.factor(data$soldes)

# Apprentissage de l'arbre de décision (régression)
tree <- rpart(
  achats ~ .,
  data = data,
  method = "anova",
  control = rpart.control(
    minsplit = 2,
    minbucket = 2,
    cp = 0.15
  )
)

# Affichage de l'arbre
rpart.plot(tree, type = 2, extra = 101, fallen.leaves = TRUE)
