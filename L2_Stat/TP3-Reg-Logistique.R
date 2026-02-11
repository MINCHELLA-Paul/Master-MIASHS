# CHAPITRE 3 - Régression logistique Exercice 2 ####

par(mfrow = c(1,1))
rm(list = ls())


# 0) Chargement des données

library(MASS)
data(Pima.tr)

df <- Pima.tr

# Variable réponse binaire Y = 1(type == "Yes"), 0 sinon
df$Y <- as.integer(df$type == "Yes")

# Covariables retenues
X_cols <- c("npreg", "glu", "bmi", "age")

# Jeu de données de travail
df_sub <- df[, c("Y", "type", X_cols)]


cat("1) Analyse descriptive des données\n")



# 1.a) Statistiques descriptives

n <- nrow(df_sub)
p <- length(X_cols)

cat(sprintf("Taille de l'échantillon n = %d\n", n))
cat(sprintf("Nombre de covariables p = %d\n\n", p))

# Proportions de classes
cat("Proportions de classes (Y):\n")
print(prop.table(table(df_sub$Y)))
cat("\n")

# Fonction utilitaire : stats descriptives + NA
desc_stats <- function(x) {
  c(
    mean = mean(x, na.rm = TRUE),
    sd   = sd(x, na.rm = TRUE),
    min  = min(x, na.rm = TRUE),
    med  = median(x, na.rm = TRUE),
    max  = max(x, na.rm = TRUE),
    empty   = sum(is.na(x))
  )
}

cat("Statistiques descriptives par covariable:\n")
desc_table <- t(sapply(df_sub[, X_cols, drop = FALSE], desc_stats))
print(round(desc_table, 3))
cat("\n")

cat("Résumé (summary) des variables:\n")
print(summary(df_sub[, c("Y", X_cols)]))
cat("\n")


# 1.b) Pairs plot par classe Y


cat("1.b) Pairs plot coloré par classe Y\n")


cols <- ifelse(df_sub$Y == 1, "#D55E00", "#0072B2")  # 1=orange, 0=bleu

pairs(
  df_sub[, X_cols],
  col = cols,
  pch = 19,
  main = "Pairs plot des covariables (couleur = classe Y)"
)

# Disposition 2x2
par(mfrow = c(2, 2))

# Couleurs (semi-transparentes pour superposition)
col0 <- rgb(0, 114, 178, maxColorValue = 255, alpha = 120)  # Y=0
col1 <- rgb(213, 94, 0,  maxColorValue = 255, alpha = 120)  # Y=1

for (var in c("npreg", "glu", "bmi", "age")) {
  
  hist(df_sub[df_sub$Y == 0, var],
       col = col0,
       freq = FALSE,
       main = paste("Histogramme de", var),
       xlab = var,
       border = "white")
  
  hist(df_sub[df_sub$Y == 1, var],
       col = col1,
       freq = FALSE,
       add = TRUE,
       border = "white")
  
  legend("topright",
         legend = c("Y = 0", "Y = 1"),
         fill = c(col0, col1),
         bty = "n",
         cex = 0.8)
}


# legend( "topright", legend = c("Y=0 (No)", "Y=1 (Yes)"),
#  col = c("#0072B2", "#D55E00"),  pch = 19,  bty = "n",cex = 0.9)


# 2) Rappels théoriques (rappel textuel, ici en console)


cat("2) Rappels théoriques (rappel console)\n")

cat("Modèle logistique : P(Y=1|X) = 1 / (1 + exp(-(w0 + sum_j wj X^(j))))\n")
cat("Hypothèses principales : (i) observations indépendantes, (ii) Y|X ~ Bernoulli(p(X)),\n")
cat("avec lien logit : log(p(X)/(1-p(X))) = w0 + sum_j wj X^(j)\n")
cat("Critère optimisé : maximum de vraisemblance (équivalent à minimiser la log-loss).\n\n")


# 3) Apprentissage du modèle dans R


cat("3) Apprentissage du modèle (glm binomial)\n")


# Modèle logistique
form <- as.formula(paste("Y ~", paste(X_cols, collapse = " + ")))
model <- glm(form, data = df_sub, family = binomial)

cat("Résumé du modèle (summary):\n")
print(summary(model))
cat("\n")

# 3.a) Coefficients estimés + IC 95%
cat("Coefficients estimés (w-hat):\n")
print(coef(model))
cat("\n")

# IC 95% (Wald, rapide et standard en TD)
ci_wald <- suppressMessages(confint.default(model))  # ±1.96*SE sur l'échelle des coefficients
colnames(ci_wald) <- c("2.5 %", "97.5 %")

cat("Intervalles de confiance à 95% (Wald):\n")
print(ci_wald)
cat("\n")

# (Optionnel) IC profilés (plus fidèles, mais parfois plus lents)
# ci_profile <- suppressMessages(confint(model))
# cat("Intervalles de confiance à 95% (profil de vraisemblance):\n")
# print(ci_profile); cat("\n")

# 3.b) Significativité : 0 appartient-il à l'IC ?
sig <- apply(ci_wald, 1, function(row) !(0 >= row[1] && 0 <= row[2]))
sig <- as.logical(sig)

sig_table <- data.frame(
  coefficient = names(coef(model)),
  estimate    = coef(model),
  CI_2.5      = ci_wald[, 1],
  CI_97.5     = ci_wald[, 2],
  significant = sig
)

cat("Significativité (au sens : 0 ∉ IC95% => significatif):\n")
print(sig_table)
cat("\n")

cat("Interprétation des signes (pour les coefficients significatifs):\n")
for (j in seq_along(sig)) {
  if (sig[j]) {
    name_j <- names(coef(model))[j]
    est_j  <- coef(model)[j]
    if (name_j == "(Intercept)") next
    if (est_j > 0) {
      cat(sprintf("- %s : coefficient positif => augmente les log-odds, donc augmente P(Y=1).\n", name_j))
    } else {
      cat(sprintf("- %s : coefficient négatif => diminue les log-odds, donc diminue P(Y=1).\n", name_j))
    }
  }
}
cat("\n")

cat("Pertinence globale (indication) :\n")
cat("- Consulter la déviance résiduelle, l'AIC, et éventuellement un test du rapport de vraisemblance.\n\n")


# 4) Prédictions sur de nouvelles observations

cat("4) Prédiction sur de nouvelles observations\n")

# Trois nouvelles observations (exemples numériques plausibles)
# Remarque : adaptez les valeurs si vous souhaitez coller à un scénario précis en TD.
X_new <- data.frame(
  npreg = c(1, 3, 6),
  glu   = c(90, 130, 170),
  bmi   = c(22, 30, 36),
  age   = c(22, 35, 50)
)

cat("Nouvelles observations X_k:\n")
print(X_new)
cat("\n")

# Probabilités prédites p_hat_k = P(Y=1|X=X_k)
p_hat <- predict(model, newdata = X_new, type = "response")

# Décision au seuil tau = 1/2
tau <- 0.5
y_hat <- as.integer(p_hat >= tau)

pred_table <- cbind(X_new, p_hat = p_hat, y_hat = y_hat)

cat(sprintf("Seuil de décision tau = %.2f\n", tau))
cat("Table des prédictions (p_hat et classe prédite):\n")
print(pred_table)
cat("\n")

cat("Interprétation:\n")
cat("- p_hat proche de 0 => faible probabilité de diabète (classe 0)\n")
cat("- p_hat proche de 1 => forte probabilité de diabète (classe 1)\n")
cat("- Décision : y_hat = 1 si p_hat >= 0.5, sinon 0\n")



# ---------- FIN Exercice 2 ----------


# EXERCICE 3 -- ReGRESSION LOGISTIQUE ####


# TD-TP Régression logistique multinomiale — Jeu de données iris
# Script R complet pour répondre à l’énoncé


# --- 0) Chargements et préparation
par(mfrow = c(1,1))
rm(list = ls())

data(iris)                 # jeu de données intégré à R
df <- iris                 # on travaille sur df
df$Species <- factor(df$Species)

# (Optionnel) ré-encoder Y en {1,2,3} en gardant la correspondance de l’énoncé
# 1 = setosa, 2 = versicolor, 3 = virginica
df$Y_num <- as.integer(df$Species)

# Variables explicatives (X^{(1)},...,X^{(4)})
X_cols <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")


# 1) Analyse descriptive
# a) Taille n et nombre de variables explicatives p
n <- nrow(df)
p <- length(X_cols)
cat("=== 1.a) Dimensions ===\n")
cat("n =", n, "\n")
cat("p =", p, "\n\n")

# b) Nature des variables + stats descriptives usuelles
cat("=== 1.b) Nature des variables ===\n")
str(df)

cat("\n=== 1.b) Nombre de NA ===\n")
print(colSums(is.na(df)))

cat("\n=== 1.b) Statistiques descriptives (covariables) ===\n")
# summary donne min, quartiles, mediane, moyenne, max
print(summary(df[, X_cols]))

# Ajout des moyennes et écarts-types (souvent demandés)
cat("\n=== Moyennes et ecarts-types (covariables) ===\n")
means <- sapply(df[, X_cols], mean, na.rm = TRUE)
sds   <- sapply(df[, X_cols], sd,   na.rm = TRUE)
print(rbind(mean = means, sd = sds))

# c) Matrice de plots 4x4 (pairs)
cat("\n=== 1.c) Matrice de plots 4x4 (pairs) ===\n")
# Couleurs par espèce
cols <- c("setosa" = "red",
          "versicolor" = "darkgreen",
          "virginica" = "blue")

# Fonction panneau diagonal : histogrammes par classe
panel.hist.byclass <- function(x, ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))  # ajustement vertical
  
  for (sp in levels(df$Species)) {
    x_sp <- x[df$Species == sp]
    h <- hist(x_sp, plot = FALSE, breaks = "FD")
    y <- h$counts / max(h$counts)   # normalisation pour superposition
    rect(h$breaks[-length(h$breaks)], 0,
         h$breaks[-1], y,
         col = adjustcolor(cols[sp], alpha.f = 0.4),
         border = NA)
  }
}

# Fonction panneau hors diagonale
panel.scatter <- function(x, y, ...) {
  points(x, y, col = cols[df$Species], pch = 19)
}

# Pairs avec histogrammes sur la diagonale
pairs(df[, X_cols],
      lower.panel = panel.scatter,
      upper.panel = panel.scatter,
      diag.panel  = panel.hist.byclass,
      main = "Pairs plot avec histogrammes par classe")

# legend("topright", legend = levels(df$Species), col = cols[levels(df$Species)],
#        pch = 19, bty = "n", cex = 0.9)
# Correspondance affichée dans la console
cat("Correspondance couleur / classe :\n")
for (sp in levels(df$Species)) {
  cat(sp, ":", cols[sp], "\n")
}

# 2) Modélisation logistique multinomiale
# On utilise nnet::multinom (régression logistique multinomiale)

if (!requireNamespace("nnet", quietly = TRUE)) {
  install.packages("nnet")
}
library(nnet)

# Ajustement : Species ~ covariables
model_multi <- multinom(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                        data = df, trace = FALSE)

cat("=== 2) Modele multinomial ajuste ===\n")
print(model_multi)

# Résumé (coefficients + erreurs standards approx.)
cat("\n=== Summary du modele multinomial ===\n")
sum_multi <- summary(model_multi)
print(sum_multi)


# 3) Probabilités prédites pour 3 individus
# Définition des 3 individus (X1, X2, X3) donnés dans l’énoncé
X_new <- data.frame(
  Sepal.Length = c(5.1, 6.0, 6.5),
  Sepal.Width  = c(3.5, 2.9, 3.0),
  Petal.Length = c(1.4, 4.5, 5.5),
  Petal.Width  = c(0.2, 1.5, 2.0)
)

rownames(X_new) <- c("X1", "X2", "X3")

cat("\n=== 3) Individus consideres ===\n")
print(X_new)

# a) Probabilités prédites pour chaque classe
proba_pred <- predict(model_multi, newdata = X_new, type = "probs")

cat("\n=== 3.a) Probabilites predites P(Y=j | X=Xk) ===\n")
print(round(proba_pred, 6))

# b) Vérifier que les probas somment à 1
cat("\n=== 3.b) Verification somme des probas (=1) ===\n")
print(rowSums(proba_pred))


# 4) Règle de décision (classe prédite)
# a) Critère : argmax_j p_hat^{(j)}
class_pred <- predict(model_multi, newdata = X_new, type = "class")

cat("\n=== 4) Classe predite (argmax des probabilites) ===\n")
print(class_pred)

# (Optionnel) afficher aussi la classe sous forme numérique 1,2,3
class_pred_num <- as.integer(class_pred)
cat("\nClasses predites sous forme numerique (1=setosa,2=versicolor,3=virginica) :\n")
print(setNames(class_pred_num, rownames(X_new)))


# ---------- FIN Exercice 3 ----------