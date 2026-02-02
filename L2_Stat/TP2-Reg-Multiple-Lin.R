# EXERCICE 3 -- ReGRESSION LINeAIRE MULTIPLE (Cholesterol) ####
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

# 2.b) Estimation de la variance du bruit : sigma^2 (au sens MCO)
# R donne directement sigma_hat via summary(model)$sigma ; on recupere sigma^2 en le mettant au carre.
sigma_hat  <- summary(model)$sigma
sigma2_hat <- sigma_hat^2
sigma_hat
sigma2_hat

# 3) Intervalles de confiance a 95% pour chaque coefficient
confint(model, level = 0.95)

# 4) Analyse des residus : centrage, histogramme, QQ-plot
res <- residuals(model)

# Centrage numerique
mean(res)

# Test de Student : moyenne des residus nulle ?
t.test(res, mu = 0)

# Histogramme des residus
hist(res,breaks = 10, main   = "Histogramme des residus",
     xlab   = "Residus",col  = "purple")

# QQ-plot des residus (diagnostic de normalite)
qqnorm(res, main = "QQ-plot des residus")
qqline(res, col = "red", lwd = 2)

# Diagnostic d'homoscedasticite : residus vs valeurs ajustees
plot(fitted(model), res, pch  = 19, xlab = "Valeurs ajustees",
     ylab = "Residus", main = "Residus vs valeurs ajustees")
abline(h = 0, lty = 2)

# 4.b) Test (optionnel) de normalite des residus : Shapiro-Wilk (n=15)... sensible aux petits échantillons.
shapiro.test(res)

# 5) Comparaison avec d'autres modeles (exemples)
# 5.a) Modele reduit : selection de variables (a discuter selon significativite)
model_red <- lm(cholesterol ~ age, data = df)
summary(model_red)

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


# ---------- FIN Exercice 3 ----------