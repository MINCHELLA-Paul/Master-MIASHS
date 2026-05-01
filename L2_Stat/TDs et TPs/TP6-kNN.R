library(mlbench)
library(class)
library(dplyr)


# Exercice 1 ####
## 1NN####
library(ggplot2)


# Données observées
df <- data.frame(
  age = c(22, 25, 28, 35, 40, 45, 30, 38),
  temps_libre = c(12, 10, 4, 6, 8, 3, 11, 9),
  Y = factor(c(1, 1, 0, 0, 0, 0, 1, 1))
)



# Nouvelles observations


new_points <- data.frame(
  age = c(27, 37, 33),
  temps_libre = c(9, 8.5, 5),
  label = c("xA", "xB", "xC")
)



# Grille pour frontière k-NN


x_range <- seq(min(df$age)-3,
               max(df$age)+3,
               length.out = 300)

y_range <- seq(min(df$temps_libre)-3,
               max(df$temps_libre)+3,
               length.out = 300)

grid <- expand.grid(
  age = x_range,
  temps_libre = y_range
)



# Prédictions 1-NN


pred <- knn(
  train = df[,c("age","temps_libre")],
  test = grid,
  cl = df$Y,
  k = 1
)

grid$pred <- pred



# Plot final


ggplot() +
  
  # régions colorées
  geom_tile(
    data = grid,
    aes(age, temps_libre, fill = pred),
    alpha = 0.25
  ) +
  
  # frontière plus épaisse
  geom_contour(
    data = grid,
    aes(age, temps_libre,
        z = as.numeric(pred)),
    breaks = 1.5,
    colour = "black",
    linewidth = 1.2
  ) +
  
  # points observés
  geom_point(
    data = df,
    aes(age, temps_libre,
        shape = Y,
        colour = Y),
    size = 3
  ) +
  
  # nouvelles observations
  geom_point(
    data = new_points,
    aes(age, temps_libre),
    shape = 8,
    size = 5,
    colour = "black"
  ) +
  
  geom_text(
    data = new_points,
    aes(age, temps_libre,
        label = label),
    nudge_x = 0.4
  ) +
  
  scale_shape_manual(
    values = c("0" = 1,
               "1" = 17)
  ) +
  
  scale_colour_manual(
    values = c("0" = "blue",
               "1" = "red")
  ) +
  
  scale_fill_manual(
    values = c("0" = "lightblue",
               "1" = "mistyrose")
  ) +
  
  labs(
    title = "Frontière de décision du plus proche voisin (k = 1)",
    x = "Âge (années)",
    y = "Temps libre (heures / semaine)",
    colour = "Classe",
    shape = "Classe",
    fill = "Classe prédite"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    panel.border = element_rect(
      colour = "black",
      fill = NA
    )
  )

## 3NN ####

x_range <- seq(min(df$age)-3,
               max(df$age)+3,
               length.out = 300)

y_range <- seq(min(df$temps_libre)-3,
               max(df$temps_libre)+3,
               length.out = 300)

grid <- expand.grid(
  age = x_range,
  temps_libre = y_range
)



pred <- knn(
  train = df[,c("age","temps_libre")],
  test = grid,
  cl = df$Y,
  k = 3
)

grid$pred <- pred



# Plot final


ggplot() +
  
  # régions colorées
  geom_tile(
    data = grid,
    aes(age, temps_libre, fill = pred),
    alpha = 0.25
  ) +
  
  # frontière plus épaisse
  geom_contour(
    data = grid,
    aes(age, temps_libre,
        z = as.numeric(pred)),
    breaks = 1.5,
    colour = "black",
    linewidth = 1.2
  ) +
  
  # points observés
  geom_point(
    data = df,
    aes(age, temps_libre,
        shape = Y,
        colour = Y),
    size = 3
  ) +
  
  # nouvelles observations
  geom_point(
    data = new_points,
    aes(age, temps_libre),
    shape = 8,
    size = 5,
    colour = "black"
  ) +
  
  geom_text(
    data = new_points,
    aes(age, temps_libre,
        label = label),
    nudge_x = 0.4
  ) +
  
  scale_shape_manual(
    values = c("0" = 1,
               "1" = 17)
  ) +
  
  scale_colour_manual(
    values = c("0" = "blue",
               "1" = "red")
  ) +
  
  scale_fill_manual(
    values = c("0" = "lightblue",
               "1" = "mistyrose")
  ) +
  
  labs(
    title = "Frontière de décision - kNN (k = 3)",
    x = "Âge (années)",
    y = "Temps libre (heures / semaine)",
    colour = "Classe",
    shape = "Classe",
    fill = "Classe prédite"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    panel.border = element_rect(
      colour = "black",
      fill = NA
    )
  )


# Exercice 2 ####

library(ggplot2)
library(class)

df <- data.frame(
  x = c(0,2,4,2,5,6),
  y = c(1,3,4,0,2,3),
  classe = factor(c("rouge","rouge","rouge",
                    "bleu","bleu","bleu"))
)

x_test <- data.frame(x = 1, y = 2)


frontiere_knn <- function(data, k){
  
  x_range <- seq(min(data$x)-1,
                 max(data$x)+1,
                 length.out = 300)
  
  y_range <- seq(min(data$y)-1,
                 max(data$y)+1,
                 length.out = 300)
  
  grid <- expand.grid(x = x_range,
                      y = y_range)
  
  grid$pred <- knn(train = data[,c("x","y")],
                   test  = grid,
                   cl    = data$classe,
                   k     = k)
  
  grid
  
}


plot_knn <- function(data, grid, title_plot,
                     add_test_point = FALSE,
                     scaled_axis = FALSE){
  
  p <- ggplot() +
    
    geom_tile(data = grid,
              aes(x, y, fill = pred),
              alpha = 0.35) +
    
    geom_contour(data = grid,
                 aes(x, y,
                     z = as.numeric(pred)),
                 breaks = 1.5,
                 colour = "purple",
                 linewidth = 1.4) +
    
    geom_point(data = data,
               aes(x, y,
                   colour = classe,
                   shape = classe),
               size = 3) +
    
    scale_colour_manual(values = c("bleu" = "red",
                                   "rouge" = "turquoise4")) +
    
    scale_shape_manual(values = c("bleu" = 15,
                                  "rouge" = 16)) +
    
    labs(title = title_plot,
         x = "x",
         y = ifelse(scaled_axis,"5y","y"),
         fill = "Classe prédite") +
    
    theme_minimal(base_size = 14) +
    
    theme(panel.border = element_rect(colour = "black",
                                      fill = NA))
  
  if(add_test_point){
    
    p <- p +
      geom_point(data = x_test,
                 aes(x,y),
                 shape = 8,
                 size = 5) +
      annotate("text",
               x = 1.2,
               y = 2,
               label = "x = (1,2)")
  }
  
  p
  
}


grid_1nn <- frontiere_knn(df, 1)

plot_knn(df,
         grid_1nn,
         "Frontière de décision – 1-NN")


df_scaled <- df
df_scaled$y <- 5 * df_scaled$y

grid_scaled <- frontiere_knn(df_scaled, 1)

plot_knn(df_scaled,
         grid_scaled,
         "Frontière 1-NN après changement d'échelle (y × 5)",
         scaled_axis = TRUE)


grid_3nn <- frontiere_knn(df, 3)

plot_knn(df,
         grid_3nn,
         "Frontière de décision – 3-NN",
         add_test_point = TRUE)



# Classification du point test


knn(
  train = df[,c("x","y")],
  test = x_test,
  cl = df$classe,
  k = 3
)



# Exercice 3 ####
library(mlbench)
library(dplyr)
library(class)
library(ggplot2)


data(PimaIndiansDiabetes)

df <- PimaIndiansDiabetes


set.seed(123)

non_diab <- df %>%
  filter(diabetes == "neg",
         glucose < 120,
         mass < 45,
         mass > 15) %>%
  sample_n(15)

diab <- df %>%
  filter(diabetes == "pos",
         glucose > 80,
         mass > 20,
         mass < 4500) %>%
  sample_n(15)

df_small <- bind_rows(non_diab, diab)

df_small <- df_small[df_small$mass <= 40, ]


df_desc <- df_small[, c("mass","glucose","diabetes")]

df_num <- df_desc[, c("mass","glucose")]


summary_table <- data.frame(
  Variable = names(df_num),
  N_OBS = sapply(df_num, function(x) sum(!is.na(x))),
  Na = sapply(df_num, function(x) sum(is.na(x))),
  Mean = sapply(df_num, mean),
  Std = sapply(df_num, sd),
  Var = sapply(df_num, var),
  Min = sapply(df_num, min),
  Med = sapply(df_num, median),
  Max = sapply(df_num, max)
)

print(summary_table)

table(df_desc$diabetes)
prop.table(table(df_desc$diabetes))


X <- df_small[, c("mass","glucose")]
Y <- df_small$diabetes


x_range <- seq(min(X$mass)-2,
               max(X$mass)+2,
               length.out = 250)

y_range <- seq(min(X$glucose)-5,
               max(X$glucose)+5,
               length.out = 250)

grid <- expand.grid(
  mass = x_range,
  glucose = y_range
)


plot_knn <- function(k){
  
  pred <- knn(
    train = X,
    test = grid,
    cl = Y,
    k = k
  )
  
  grid_df <- data.frame(grid, pred)
  
  
  ggplot() +
    
    geom_tile(
      data = grid_df,
      aes(mass, glucose, fill = pred),
      alpha = 0.35
    ) +
    
    geom_contour(
      data = grid_df,
      aes(mass, glucose,
          z = as.numeric(pred)),
      bins = 1,
      colour = "purple",
      linewidth = 1.21
    ) +
    
    geom_point(
      data = df_small %>% filter(diabetes == "neg"),
      aes(mass, glucose,
          shape = "Non diabétique",
          colour = "Non diabétique"),
      size = 3
    ) +
    
    geom_point(
      data = df_small %>% filter(diabetes == "pos"),
      aes(mass, glucose,
          shape = "Diabétique",
          colour = "Diabétique"),
      size = 3
    ) +
    
    scale_colour_manual(
      values = c("Non diabétique" = "blue",
                 "Diabétique" = "red"),
      name = "Observations"
    ) +
    
    scale_fill_manual(
      values = c("neg" = "#A6DBEF",
                 "pos" = "#F3A6A6"),
      name = "Région prédite"
    ) +
    
    scale_shape_manual(
      values = c("Non diabétique" = 4,
                 "Diabétique" = 12),
      name = "Observations"
    ) +
    
    labs(
      title = paste("Frontière de décision k-NN (k =", k, ")"),
      x = "Indice de masse corporelle (BMI)",
      y = "Taux de glucose"
    ) +
    
    theme_minimal(base_size = 14) +
    
    theme(
      legend.position = "right",
      panel.border = element_rect(
        colour = "black",
        fill = NA
      )
    )
  
}


par(mfrow = c(1,2))

print(plot_knn(1))
print(plot_knn(3))

par(mfrow = c(1,1))



# Exercice 5 ####
library(class)
library(ggplot2)
library(dplyr)



# 1. Données


df <- data.frame(
  X1 = c(12,15,18,22,25,28,30,20),
  X2 = c(18,22,25,30,35,38,40,28),
  Y  = c(0,0,0,1,1,1,1,0)
)

df$Y_factor <- factor(df$Y)

x_new <- data.frame(X1 = 20, X2 = 30)



# 2. Représentation graphique


ggplot(df, aes(X1, X2, colour = Y_factor)) +
  geom_point(size = 4) +
  geom_point(data = x_new,
             aes(X1, X2),
             colour = "black",
             shape = 8,
             size = 5) +
  labs(
    title = "Nuage de points + observation à prédire",
    x = expression(X^{(1)}),
    y = expression(X^{(2)})
  ) +
  theme_minimal()



# 3. Calcul des distances


distance <- function(a, b) {
  sqrt((a[1]-b[1])^2 + (a[2]-b[2])^2)
}

df$distance <- apply(
  df[,1:2],
  1,
  function(z) distance(z, x_new)
)

df <- df %>% arrange(distance)

print(df)



# 4. Classification 3-NN


k3_neighbors <- df[1:3,]

print(k3_neighbors)

table(k3_neighbors$Y)

pred_3NN <- names(which.max(table(k3_neighbors$Y)))

cat("Classe prédite (3-NN) :", pred_3NN, "\n")



# 5. Classification 5-NN


k5_neighbors <- df[1:5,]

print(k5_neighbors)

table(k5_neighbors$Y)

pred_5NN <- names(which.max(table(k5_neighbors$Y)))

cat("Classe prédite (5-NN) :", pred_5NN, "\n")



# 6. Méthode epsilon-voisins


epsilon <- 6

eps_neighbors <- df %>%
  filter(distance <= epsilon)

print(eps_neighbors)

table(eps_neighbors$Y)

pred_eps <- names(which.max(table(eps_neighbors$Y)))

cat("Classe prédite (epsilon = 6) :", pred_eps, "\n")



# 7. Influence de epsilon


epsilon_values <- c(3, 10)

for(eps in epsilon_values){
  
  cat("\n----------------------------\n")
  cat("epsilon =", eps, "\n")
  
  neigh <- df %>%
    filter(distance <= eps)
  
  print(neigh)
  
  if(nrow(neigh) == 0){
    
    cat("Aucun voisin disponible\n")
    
  } else {
    
    pred <- names(which.max(table(neigh$Y)))
    
    cat("Classe prédite :", pred, "\n")
    
  }
  
}



# 8. Illustration graphique epsilon-voisins


epsilon <- 6


theta <- seq(0, 2*pi, length.out = 200)

circle_df <- data.frame(
  x = x_new$X1 + epsilon * cos(theta),
  y = x_new$X2 + epsilon * sin(theta)
)

ggplot(df, aes(X1, X2)) +
  
  geom_point(aes(colour = Y_factor),
             size = 4) +
  
  geom_point(data = x_new,
             aes(X1, X2),
             shape = 8,
             size = 5) +
  
  geom_path(
    data = circle_df,
    aes(x, y),
    colour = "black",
    linetype = "dashed"
  ) +
  
  labs(
    title = paste("Voisinage epsilon =", epsilon),
    x = expression(X^{(1)}),
    y = expression(X^{(2)})
  ) +
  
  theme_minimal()



# 9. Comparaison automatique k-NN vs epsilon-voisins


compare_method <- function(k, eps){
  
  pred_knn <- knn(
    train = df[,1:2],
    test  = x_new,
    cl    = df$Y_factor,
    k     = k
  )
  
  neigh_eps <- df %>%
    filter(distance <= eps)
  
  pred_eps <- ifelse(
    nrow(neigh_eps) == 0,
    NA,
    names(which.max(table(neigh_eps$Y)))
  )
  
  cat("\nComparaison méthodes\n")
  cat("k =", k, "->", pred_knn, "\n")
  cat("epsilon =", eps, "->", pred_eps, "\n")
  
}

compare_method(3,6)