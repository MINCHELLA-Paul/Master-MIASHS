# CORRECTION -- EXERCICE 1 : RÉGRESSION LINEAIRE SIMPLE ####
# 1. Chargement du jeu de donnees
data(cars)


# 2. Construction explicite du data frame de travail
df <- data.frame(
  speed = cars$speed,
  dist  = cars$dist
)


# 3. Statistique descriptive du jeu de donnees
str(df)
summary(df)

range(df$speed)
range(df$dist)
sd(df$speed)
sd(df$dist)
IQR(df$speed)
IQR(df$dist)


# 4. Histogrammes des variables
par(mfrow = c(1,2))
hist(df$speed,
     breaks = 10,
     main   = "Histogramme de la vitesse",
     xlab   = "Vitesse (mph)",
     col    = "lightgray")

hist(df$dist,
     breaks = 10,
     main   = "Histogramme de la distance de freinage",
     xlab   = "Distance de freinage (ft)",
     col    = "darkgreen")


# 5. Nuage de points : distance en fonction de la vitesse
par(mfrow = c(1,1))
plot(df$speed, df$dist,
     pch  = 19,
     col  = "darkgreen",
     xlab = "Vitesse (mph)",
     ylab = "Distance de freinage (ft)",
     main = "Distance de freinage en fonction de la vitesse")


# 6. Ajustement du modele de regression lineaire
model <- lm(dist ~ speed, data = df)


# 7. Resume global du modele
summary(model)

# Intervalles de confiance à 95 % des coefficients
confint(model, level = 0.95)

# 8. Visualisation : points, droite de regression et residus
# Droite de regression
abline(model, col = "red", lwd = 2)

# Residus (distances verticales a la droite)
segments(x0 = df$speed,
         y0 = fitted(model),
         x1 = df$speed,
         y1 = df$dist,
         col = "purple",
         lty = 2)

# Legende
legend("topleft",
       legend = c("Observations", "Droite de regression", "Residus"),
       col    = c("darkgreen", "red", "purple"),
       pch    = c(19, NA, NA),
       lty    = c(NA, 1, 2),
       lwd    = c(NA, 2, 1),
       bty    = "n")


# 9. Analyse des residus
res <- residuals(model)

hist(res,
     breaks = 10,
     main   = "Histogramme des residus",
     xlab   = "Residus",
     col    = "purple")

# Verification du centrage des residus
mean(res)

# Test de Student : moyenne des résidus nulle ?
t.test(res, mu = 0)

# Diagnostic pour vérifier la normalité: QQ-plot
qqnorm(res, main = "QQ-plot des résidus")
qqline(res, col = "red", lwd = 2)


# ---------- FIN Exercice 1 ----------



# CORRECTION -- EXERCICE 2 : RÉGRESSION LINÉAIRE SIMPLE ####
# Jeu de données : faithful

# 1. Chargement du jeu de données
data(faithful)

# Construction explicite du data frame de travail
df <- data.frame(
  waiting   = faithful$waiting,
  eruptions = faithful$eruptions
)

head(df)

# 2. Statistique descriptive du jeu de données
str(df)
summary(df)

# Histogrammes des variables
par(mfrow = c(1, 2))

hist(df$waiting,
     breaks = 15,
     main   = "Histogramme du temps d'attente",
     xlab   = "Temps d'attente (min)",
     col    = "lightgray")

hist(df$eruptions,
     breaks = 15,
     main   = "Histogramme de la durée des éruptions",
     xlab   = "Durée des éruptions (min)",
     col    = "darkgreen")

# 3. Visualisation bivariée : scatter plot
par(mfrow = c(1, 1))

plot(df$waiting, df$eruptions,
     pch  = 19,
     col  = "darkgreen",
     xlab = "Temps d'attente (min)",
     ylab = "Durée des éruptions (min)",
     main = "Durée des éruptions en fonction du temps d'attente")

# 4. Ajustement du modèle de régression linéaire
model <- lm(eruptions ~ waiting, data = df)

# Résumé global du modèle
summary(model)

# 5. Intervalles de confiance à 95 % des coefficients
confint(model, level = 0.95)

# 6. Visualisation : points, droite de régression et résidus
plot(df$waiting, df$eruptions,
     pch  = 19,
     col  = "darkgreen",
     xlab = "Temps d'attente (min)",
     ylab = "Durée des éruptions (min)",
     main = "Régression linéaire : eruptions ~ waiting")

# Droite de régression
abline(model, col = "red", lwd = 2)

# Résidus (distances verticales à la droite)
segments(x0 = df$waiting,
         y0 = fitted(model),
         x1 = df$waiting,
         y1 = df$eruptions,
         col = "purple",
         lty = 2)

# Légende
legend("topleft",
       legend = c("Observations", "Droite de régression", "Résidus"),
       col    = c("darkgreen", "red", "purple"),
       pch    = c(19, NA, NA),
       lty    = c(NA, 1, 2),
       lwd    = c(NA, 2, 1),
       bty    = "n")

# 7. Analyse des résidus
res <- residuals(model)

# Histogramme des résidus
hist(res,
     breaks = 15,
     main   = "Histogramme des résidus",
     xlab   = "Résidus",
     col    = "purple")

# Vérification du centrage des résidus
mean(res)

# Test de Student : moyenne des résidus nulle ?
t.test(res, mu = 0)

# Diagnostic de normalité : QQ-plot
qqnorm(res, main = "QQ-plot des résidus")
qqline(res, col = "red", lwd = 2)

# ---------- FIN Exercice 2 ----------