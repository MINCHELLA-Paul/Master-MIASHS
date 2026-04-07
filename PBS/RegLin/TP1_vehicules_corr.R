# TD 1 - Regression lineaire

# install.packages('corrplot') # Si besoin
library(corrplot)

# 1. Lecture des donnees ####

setwd("~/Desktop/TRAVAIL/EPITA/2025_03 EPITA4 PBS2/STATISTIQUE/TP1 RegLin PBS2/")
df <- read.csv("Vehicules_TD_2023.csv", 
               sep = ";", dec = ".", header = TRUE, 
               stringsAsFactors = FALSE)

# 2. Examen des donnees ####

## Statistiques descriptives ####
str(df)
dim(df)
names(df)
head(df)
summary(df)
sapply(df[, sapply(df, is.numeric)], sd, na.rm = TRUE)

## Valeurs manquantes ####
colSums(is.na(df))
sum(is.na(df))

# Variables numériques et qualitatives
num_cols <- names(df)[sapply(df, is.numeric)]
cat_cols <- names(df)[!sapply(df, is.numeric)]



## Histogramme pour chaque variable numerique ####
# Variables explicatives
vars_features <- setdiff(num_cols, "PRIX")
par(mfrow = c(2, 3))
for (v in vars_features) {
  hist(df[[v]], main = paste("Histogramme de", v), 
       xlab = v, col = "lightblue")
}

# Variables a modeliser
par(mfrow = c(1, 1))
hist(df$PRIX, main = "Histogramme du PRIX",xlab = 'PRIX', col = "#735EF1")

## Barchart variable categorielle ####

par(mfrow = c(1,2), mar = c(4,4,2,1))

freq <- table(df[[cat_cols]])
pct <- round(prop.table(freq)*100, 1)

# Barplot
barplot(freq,
        main = paste("Diagramme en barres de", cat_cols),
        xlab = cat_cols,
        col = "pink")

# Camembert avec pourcentages
pie(freq,
    labels = paste(names(freq), pct, "%"),
    main = paste("Diagramme circulaire de", cat_cols),
    col = rainbow(length(freq)))

par(mfrow = c(1,1))

## Pairplots ####
pairs(df[, num_cols], main = "Pairplot")

## Boxplot pour chaque variable numerique ####

# Variables numeriques et qualitatives
num_cols <- names(df)[sapply(df, is.numeric)]
cat_cols <- names(df)[!sapply(df, is.numeric)]

par(mfrow = c(3, 2))
for (v in vars_features) {
  boxplot(df[[v]], main = paste("Boxplot de", v), 
          horizontal = TRUE, col='lightblue')
}

par(mfrow = c(1, 1))
boxplot(df$PRIX, main = "Boxplot du PRIX", 
        horizontal = TRUE, col = "#735EF1")


## Matrice de correlation
all_corr <- cor(df[num_cols])
all_corr
corrplot(all_corr)

# 3. Centre de gravite par GAMME ####

df[[cat_cols]] <- as.factor(df[[cat_cols]])
print(table(df[[cat_cols]]))

plot_plan <- function(xvar, yvar, df, cat_cols) {
  
  if (!is.null(cat_cols) && all(c(xvar, yvar) %in% names(df))) {
    
    plot(
      df[[xvar]], df[[yvar]],
      col = as.integer(df[[cat_cols]]),
      pch = 19,
      xlab = xvar,
      ylab = yvar,
      main = paste(yvar, "vs", xvar, "par", cat_cols)
    )
    
    legend(
      "topright",
      legend = levels(df[[cat_cols]]),
      col = seq_along(levels(df[[cat_cols]])),
      pch = 19
    )
    
    centres <- aggregate(
      df[, c(xvar, yvar)],
      by = list(Categorie = df[[cat_cols]]),
      FUN = mean,
      na.rm = TRUE
    )
    
    points(
      centres[[xvar]],
      centres[[yvar]],
      pch = 8,
      cex = 2,
      lwd = 2,
      col = seq_len(nrow(centres))
    )
    
    text(
      centres[[xvar]],
      centres[[yvar]],
      labels = centres$Categorie,
      pos = 3
    )
  }
}

par(mfrow = c(1,2))
plot_plan("CYL", "PUIS", df, cat_cols)
plot_plan("PUIS", "POIDS", df, cat_cols)
par(mfrow = c(1,1))

# 4. Regression lineaire ####
## Instance du modele ####
Modele <- lm(PRIX ~ CYL + PUIS + LON + LAR + POIDS + VITESSE, data = df)
summary(Modele)

## Retrouver les coefficients ajustes du modele ####
X <- model.matrix(Modele)
Y <- df$PRIX
beta_chap <- solve(t(X) %*% X) %*% t(X) %*% Y; beta_chap

## Valeurs ajustees VS valeurs reelles ####
y_hat <- fitted(Modele)

plot(Y, y_hat,
     xlab = "PRIX observe",
     ylab = "PRIX prevu",
     pch = 19,
     main = "PRIX observe vs PRIX prevu")
abline(0, 1, col = "red", lwd = 2)

## Etude des residues ####
res <- residuals(Modele)

# Vif coup doeil
head(Y - y_hat)
head(res)

# Histogramme
par(mfrow = c(1, 2))
hist(res, breaks = 15, main = "Histogramme des residus",
     xlab = "Residus", col = "purple")
qqnorm(res)
qqline(res, col = "red", lwd = 2)
par(mfrow = c(1, 1))

# Test de la normalite
shapiro.test(res)

# Test de centralite 
t.test(res)

# 5. (In)stabilite des coefficients ####
# Variables explicatives du modèle reduit
vars_model <- c("CYL", "PUIS", "LON", "LAR", "VITESSE")

# Formule du modèle
formule <- as.formula(
  paste("PRIX ~", paste(vars_model, collapse = " + "))
)

# Indices supprimes (3 experiences differentes)
rows_removed <- c(1, 5, 10)

# Tableau pour stocker les coefficients
coef_table <- matrix(NA,
                     nrow = length(rows_removed),
                     ncol = length(coef(lm(formule, data = df))))

colnames(coef_table) <- names(coef(lm(formule, data = df)))
rownames(coef_table) <- paste("Sans ligne", rows_removed)

# Boucle : suppression d'une ligne puis recalcul modèle
for (i in seq_along(rows_removed)) {
  df_tmp <- df[-rows_removed[i], ]
  modele_tmp <- lm(formule, data = df_tmp)
  coef_table[i, ] <- coef(modele_tmp)
}


coef_initial <- coef(lm(formule, data = df))
coef_table <- rbind(
  "Modele complet" = coef_initial,
  coef_table
)
# Tableau comparatif final
coef_table


# 6. Significativite des coefficients ####
# On revient sur le summary
summary(Modele)

# Une autre approche : les intervalles de confiance
confint(Modele)

# Un modele reduit avec uniquement les variables significatives
Modele_reduit <- lm(PRIX ~ CYL + PUIS + LON + LAR + VITESSE, data = df)
summary(Modele_reduit)

# 7. Train-test split ####

# Garantir la reproductabilite à chaque relance du code
set.seed(777) 

# Split
n <- nrow(df)
id_train <- sample(1:n, size = round(0.7*n))
df_train <- df[id_train, ]
df_test  <- df[-id_train, ]

# Apprentissage sur le jeu d'entrainement
Modele_train <- lm(PRIX ~ CYL + PUIS + LON + LAR + VITESSE,
                   data = df_train)
summary(Modele_train)

# Inference sur le jeu de test
y_test <- df_test$PRIX
y_pred <- predict(Modele_train, newdata = df_test)

# Performances sur le jeu de test
mean((y_test - y_pred)^2)

# Graphe PRIX observe vs PRIX predit (jeu de test)
plot(
  y_test, y_pred,
  xlab = "PRIX observe (test)",
  ylab = "PRIX predit",
  main = "PRIX observe vs PRIX predit (jeu de test)",
  pch = 19,
  col = "#735EF1"
)

# Droite ideale : prediction parfaite y=x
abline(0, 1, col = "red", lwd = 2)

# R2 du test
R2_test <- 1 - sum((y_test - y_pred)^2) /
  sum((y_test - mean(y_test))^2)
