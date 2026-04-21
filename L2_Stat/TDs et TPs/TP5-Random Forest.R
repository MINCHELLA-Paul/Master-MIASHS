# Exerice 1 ####

# TD–TP : Arbres de décision et forêts aléatoires
# Jeu de données : mtcars (natif à R)

library(rpart); library(rpart.plot); library(ranger)
data(mtcars)
par(mfrow = c(1,2))

# Variable réponse : boîte de vitesse
mtcars$am <- factor(mtcars$am, levels = c(0, 1))


## 1) Train / test split (première seed) ####
set.seed(777)

n <- nrow(mtcars)
idx_train <- sample(1:n, size = 0.5 * n)

train <- mtcars[idx_train, ]
test  <- mtcars[-idx_train, ]


## 2) Arbre de décision (classification) ####
tree1 <- rpart(
  am ~ mpg + cyl + disp + hp + wt + qsec,
  data = train,
  method = "class",
  control = rpart.control(cp = 0, minsplit = 2)
)

rpart.plot(
  tree1,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Arbre de décision - Seed 777"
)


## 3) Performances arbre (train / test) ####
pred_train_tree1 <- predict(tree1, train, type = "class")
pred_test_tree1  <- predict(tree1, test,  type = "class")

acc_train_tree1 <- mean(pred_train_tree1 == train$am)
acc_test_tree1  <- mean(pred_test_tree1  == test$am)

cat("Arbre (seed 777) - Accuracy train :", acc_train_tree1, "\n")
cat("Arbre (seed 777) - Accuracy test  :", acc_test_tree1,  "\n\n")


## 4) Illustration du sur-apprentissage (nouveau split) ####
set.seed(42)

idx_train2 <- sample(1:n, size = 0.5 * n)

train2 <- mtcars[idx_train2, ]
test2  <- mtcars[-idx_train2, ]

tree2 <- rpart(
  am ~ mpg + cyl + disp + hp + wt + qsec,
  data = train2,
  method = "class",
  control = rpart.control(cp = 0, minsplit = 2)
)

rpart.plot(
  tree2,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Arbre de décision - Seed 42"
)

pred_train_tree2 <- predict(tree2, train2, type = "class")
pred_test_tree2  <- predict(tree2, test2,  type = "class")

acc_train_tree2 <- mean(pred_train_tree2 == train2$am)
acc_test_tree2  <- mean(pred_test_tree2  == test2$am)

cat("Arbre (seed 42) - Accuracy train :", acc_train_tree2, "\n")
cat("Arbre (seed 42) - Accuracy test  :", acc_test_tree2,  "\n\n")


## 5) Random Forest (bootstrap + agrégation) ####
set.seed(777)

rf <- ranger(
  am ~ mpg + cyl + disp + hp + wt + qsec,
  data = train,
  num.trees = 500,
  mtry = 3,
  importance = "impurity",
  probability = FALSE
)

pred_train_rf <- predict(rf, train)$predictions
pred_test_rf  <- predict(rf, test)$predictions

acc_train_rf <- mean(pred_train_rf == train$am)
acc_test_rf  <- mean(pred_test_rf  == test$am)

cat("Random Forest - Accuracy train :", acc_train_rf, "\n")
cat("Random Forest - Accuracy test  :", acc_test_rf,  "\n\n")


# Random Forest (seed 42)
set.seed(42)

rf_42 <- ranger(
  am ~ mpg + cyl + disp + hp + wt + qsec,
  data = train,
  num.trees = 500,
  mtry = 3,
  importance = "impurity",
  probability = FALSE
)

# Predictions
pred_train_rf_42 <- predict(rf_42, train)$predictions
pred_test_rf_42  <- predict(rf_42, test)$predictions

# Accuracy
acc_train_rf_42 <- mean(pred_train_rf_42 == train$am)
acc_test_rf_42  <- mean(pred_test_rf_42  == test$am)

# Print results
cat("Random Forest (seed 42) - Accuracy train :", acc_train_rf_42, "\n")
cat("Random Forest (seed 42) - Accuracy test  :", acc_test_rf_42,  "\n\n")


## 6) Importance des variables ####
rf$variable.importance

par(mfrow = c(1,1))