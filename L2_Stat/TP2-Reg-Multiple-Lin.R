# EXERCICE 1 -- ReGRESSION LINeAIRE MULTIPLE ####
### 1) Définition manuelle du jeu de données

df <- data.frame(
  X1 = 100*c(11.2, 8.0, 9.3, 5.3, 10.2, 7.4, 3.8, 4.1, 2.1, 6.9),
  X2 = c(800, 1800, 1300, 1100, 2700, 500, 200, 2200, 1550, 2500)/10000,
  Y  = c(19, 16, 17, 12, 18, 15, 9, 11, 8, 14)
)

# Vérification
cat(paste("--- Nombre d'observations :", 
          dim(df)[1],
          "\n--- Nombre de colonnes    :", 
          dim(df)[2]))



### 2) Statistiques descriptives (avec écart-type)

desc_stats <- data.frame(
  Mean = sapply(df, mean),
  SD   = sapply(df, sd),
  Min  = sapply(df, min),
  Max  = sapply(df, max)
)

print("Statistiques descriptives :")
print(round(desc_stats, 3))


### 3) Premier modèle : régression linéaire multiple

model_full <- lm(Y ~ X1 + X2, data = df)

# Résumé du modèle
print("Résumé du modèle multiple :")
summary(model_full)

# Critères d'information
cat("AIC (modèle multiple) :", AIC(model_full), "\n")
cat("BIC (modèle multiple) :", BIC(model_full), "\n")

# Intervalles de confiance à 95 %
ci_full <- confint(model_full)
print("Intervalles de confiance (modèle multiple) :")
print(ci_full[c("X1", "X2"), ])


### 4) Second modèle : modèle réduit (sans X2)

model_reduced <- lm(Y ~ X1, data = df)

# Résumé du modèle
print("Résumé du modèle réduit :")
summary(model_reduced)

# Critères d'information
cat("AIC (modèle réduit) :", AIC(model_reduced), "\n")
cat("BIC (modèle réduit) :", BIC(model_reduced), "\n")

# Intervalles de confiance
ci_reduced <- confint(model_reduced)
print("Intervalles de confiance (modèle réduit) :")
print(ci_reduced["X1", , drop = FALSE])


# ---------- FIN Exercice 1 ----------


# EXERCICE 4 -- ReGRESSION LINeAIRE MULTIPLE (Cholesterol) ####
# Y  : Cholesterol (ml/100ml)
# X1 : Poids (kg)
# X2 : Âge (ans)
# X3 : Taille (cm)


# 1) Construction explicite du data frame df
df <- data.frame(
  cholesterol = c(354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395),
  weight      = c( 84,  73,  65,  70,  76,  69,  63,  72,  79,  75,  47,  89,  65,  57,  59),
  age         = c( 46,  20,  52,  30,  57,  25,  28,  36,  57,  44,  24,  31,  52,  23,  60),
  height      = c(180, 190, 160, 155, 165, 170, 175, 180, 150, 165, 160, 165, 165, 170, 175)
)

# Appréhenson du jeu de données
head(df)
str(df)
summary(df)

rbind(
  Mean = round(sapply(df, mean, na.rm = TRUE),3),
  Std  = round(sapply(df, sd,   na.rm = TRUE),3),
  N    = round(sapply(df, function(x) sum(!is.na(x))),3)
)

# 2) Ajustement du modele lineaire multiple : Y ~ X1 + X2 + X3
model <- lm(cholesterol ~ weight + age + height, data = df)

# Resume global du modele : coefficients, tests t, R^2, test de Fisher, etc.
summary(model)

# Critères
cat("AIC :", AIC(model), "\n")
cat("BIC :", BIC(model), "\n")


# 2.b) Estimation de la variance du bruit : sigma^2 (au sens MCO)
# R donne directement sigma_hat via summary(model)$sigma ; 
# on recupere sigma^2 en l'elevant au carre.
sigma_hat  <- summary(model)$sigma
sigma2_hat <- sigma_hat^2
sigma_hat
sigma2_hat

# 3) Intervalles de confiance a 95% pour chaque coefficient
confint(model, level = 0.95)

# 4) Analyse des residus : centrage, histogramme, QQ-plot
res <- residuals(model)

# install.packages("lmtest")
# Tester l'indépendance des résidus
library(lmtest)
dwtest(model)


# Centrage numerique
mean(res)

# Test de Student : moyenne des residus nulle ?
t.test(res, mu = 0)


par(mfrow = c(1,3))

# Histogramme des residus
hist(res,breaks = 10, #probability = TRUE,
     main   = "Histogramme des residus",
     xlab   = "Residus",col  = "purple")

# Estimation de densite par kernel
#dens <- density(res)
#lines(dens, col = "darkblue", lwd = 2)


# QQ-plot des residus (diagnostic de normalite)
qqnorm(res, main = "QQ-plot des residus")
qqline(res, col = "red", lwd = 2)

# Diagnostic d'homoscedasticite : residus vs valeurs ajustees
plot(fitted(model), res,
     pch = 19,
     xlab = expression(hat(Y)[i]~~"(valeurs ajustees)"),
     ylab = expression(hat(epsilon)[i]~~"(residus)"),
     main = expression("Residus vs valeurs ajustees"~~(hat(Y)[i]~","~hat(epsilon)[i])))
abline(h = 0, lty = 2)

# 4.b) Test (optionnel) de normalite des residus : Shapiro-Wilk (n=15)... sensible aux petits échantillons.
shapiro.test(res)

# 5) Comparaison avec d'autres modeles (exemples)
# 5.a) Modele reduit : selection de variables (a discuter selon significativite)
model_red <- lm(cholesterol ~ age, data = df)
summary(model_red)

confint(model_red, level = 0.95)

res_red <- residuals(model_red)
t.test(res_red, mu = 0)
shapiro.test(res_red)
dwtest(model_red)

# Extraction des indicateurs globaux
R2_red      <- summary(model_red)$r.squared
R2adj_red   <- summary(model_red)$adj.r.squared
AIC_red     <- AIC(model_red)
BIC_red     <- BIC(model_red)

# Affichage propre
cat("=== Modele reduit : cholesterol ~ age ===\n")
cat("R^2              :", R2_red, "\n")
cat("R^2 ajuste       :", R2adj_red, "\n")
cat("AIC              :", AIC_red, "\n")
cat("BIC              :", BIC_red, "\n")


# 5.c) Exemple de modele avec interaction (a discuter)
model_int <- lm(cholesterol ~ weight + age + height + weight:age, data = df)
summary(model_int)

# 5.c) Comparaison par ANOVA (si modeles emboîtes)
anova(model_red, model)

# Comparaison synthetique des modeles : R^2, R^2 ajuste, AIC, BIC
models_comp <- data.frame(
  Model = c("Complet", "Reduit (age)", "Interaction"),
  R2 = c(
    summary(model)$r.squared, summary(model_red)$r.squared,
    summary(model_int)$r.squared
  ),
  R2_adj = c(
    summary(model)$adj.r.squared, summary(model_red)$adj.r.squared,
    summary(model_int)$adj.r.squared
  ),
  AIC = c(
    AIC(model),
    AIC(model_red),
    AIC(model_int)
  ),
  BIC = c(
    BIC(model),
    BIC(model_red),
    BIC(model_int)
  )
)

# Arrondi et print
models_comp[, -1] <- round(models_comp[, -1], 3); models_comp


# ---------- FIN Exercice 4 ----------

par(mfrow = c(1,1))
