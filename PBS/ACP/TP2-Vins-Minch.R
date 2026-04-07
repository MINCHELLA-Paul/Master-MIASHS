# TP2 ACP – Analyse en composantes principales des vins
# install.packages("FactoMineR", type = "binary")

# Packages ####
library(FactoMineR)
library(corrplot)

# Partie I ####
# Analyse exploratoire du jeu de donnees
## Question 1 ####
# Lecture des donnees

### a) Importation du fichier ####
setwd("~/Desktop/TRAVAIL/EPITA/2025_03 EPITA4 PBS2/STATISTIQUE/TP2 ACP")
df <- read.csv(
  "vins.csv",
  sep = ";",
  dec = ",",
  header = TRUE,
  row.names = 1,
  stringsAsFactors = FALSE
)

head(df)


### b) Nombre d'observations et variables ####
nrow(df)
ncol(df)
dim(df)

### c) Structure du tableau ####
str(df)
summary(df)


## Question 2 ####
# Variables qualitatives et quantitatives
### a) Identification ####
vars_quali <- df[, 1:2]
vars_quanti <- df[, -c(1, 2)]


### b) Statistiques descriptives ####
summary(vars_quanti)
sapply(vars_quanti, sd)


### c) Discussion standardisation ####
# -> justifie scale.unit = TRUE dans PCA()


## Question 3 ####
# Exploration initiale
### a) Valeurs manquantes ####

sum(is.na(df))


### b) Pertinence ACP ####
# Dataset adapte à une ACP (nombreuses variables quantitatives correlees)


## Question 4 ####
# Description des variables
# Interpretation qualitative 


## Question 5 ####
# Matrice de correlation
corrplot(
  cor(vars_quanti),
  method = "circle",
  type = "upper",
  tl.col = "black",
  tl.cex = 0.8
)




# Partie II ####
# Analyse en composantes principales
## Question 1 ####
# Mise en oeuvre de l'ACP
Vins.PCA <- PCA(
  df,
  scale.unit = TRUE,
  quali.sup = 1:2,
  graph = TRUE
)


## Question 2 ####
# Cercle des correlations
### Coordonnees des variables ####
Vins.PCA$var$coord


## Question 3 ####
# Coordonnees factorielles des variables
Vins.PCA$var$coord

## Question 4 ####
# Plan principal des individus
# Graphique genere automatiquement avec graph = TRUE


## Question 5 ####
# Coordonnees factorielles des individus
### a) Coordonnees ####
Vins.PCA$ind$coord


### d) Coordonnees sur les 5 premiers axes ####
Vins.PCA$ind$coord[, 1:5]


## Question 6 ####
# Valeurs propres
Vins.PCA$eig


## Question 7 ####
# Scree plot
eig_values <- Vins.PCA$eig[, 2]
eig_cum <- Vins.PCA$eig[, 3]

plot(
  eig_values,
  type = "b",
  col = "blue",
  xlab = "Composantes principales",
  ylab = "Pourcentage de variance expliquee",
  ylim = c(0, 100),
  main = "Scree plot normalise de l'ACP"
)

lines(eig_cum, type = "b", col = "red")

legend(
  "right",
  legend = c("Variance expliquee", "Variance cumulee"),
  col = c("blue", "red"),
  lty = 1,
  pch = 19
)

grid()




## Question 8 ####
# Lecture summary()
summary(Vins.PCA)




## Question 9 ####
# Individus et variables structurant l’axe 1
### a) Individus les plus contributifs ####
top_ind <- sort(
  Vins.PCA$ind$contrib[, 1],
  decreasing = TRUE
)
head(top_ind, 5)



### b) Variables les plus contributives ####
top_var <- sort(
  Vins.PCA$var$contrib[, 1],
  decreasing = TRUE
)
head(top_var, 5)


# Partie III ####
# Reduction de dimension
## Question 1 ####
# Nombre de composantes pour 95%
var_cum <- Vins.PCA$eig[, 3]
d_95 <- which(var_cum >= 95)[1]
cat(paste(
  "Nbr minimal de composantes pour 95% de variance:",
  d_95,
  "| Variance =",
  round(var_cum[d_95], 2),
  "%."
))


## Question 2 ####
# Interêt en machine learning
# Relation contribution \propto coordonnee²
plot(
  Vins.PCA$ind$coord[, 1]^2,
  Vins.PCA$ind$contrib[, 1],
  pch = 19,
  xlab = expression((Z[i]^1)^2),
  ylab = "Contribution à Dim 1 (%)",
  main = 'Lien entre coordonnee au carree et contrib - Axe 1'
)

n <- nrow(vars_quanti)
lambda1 <- Vins.PCA$eig[1, 1]

coef_theorique <- 100 / (n * lambda1)

abline(
  a = 0,
  b = coef_theorique,
  col = "red",
  lty = 2,
  lwd = 2
)


legend(
  "bottomright",
  legend = c("Individus", "Relation théorique : contribution \\propto coordonnée²"),
  col = c("black", "red"),
  pch = c(19, NA),
  lty = c(NA, 2),
  lwd = c(NA, 2),
  bty = "n"
)