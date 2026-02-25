##############################################
# Heart Disease Prediction - Choose Your Own #
# HarvardX PH125.9x Data Science Capstone   #
# Author: Jtest12324                         #
# Date: 2025                                 #
##############################################

# Install required packages if not already installed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ggplot2)
library(corrplot)

##############################################
# 1. Data Loading
##############################################

# Download the UCI Heart Disease dataset (Cleveland)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_data <- read.csv(url, header = FALSE, na.strings = "?")

# Assign meaningful column names
colnames(heart_data) <- c("age", "sex", "cp", "trestbps", "chol",
                          "fbs", "restecg", "thalach", "exang",
                          "oldpeak", "slope", "ca", "thal", "target")

cat("Dataset dimensions:", dim(heart_data), "\n")
cat("Missing values:", sum(is.na(heart_data)), "\n")

##############################################
# 2. Data Cleaning and Preprocessing
##############################################

# Remove rows with missing values
heart_data <- na.omit(heart_data)
cat("After removing NA - rows:", nrow(heart_data), "\n")

# Convert target to binary (0 = no disease, 1 = disease)
heart_data$target <- ifelse(heart_data$target > 0, 1, 0)

# Convert categorical variables to factors
heart_data <- heart_data %>%
  mutate(
    target   = factor(target, levels = c(0,1), labels = c("No","Yes")),
    sex      = factor(sex),
    cp       = factor(cp),
    fbs      = factor(fbs),
    restecg  = factor(restecg),
    exang    = factor(exang),
    slope    = factor(slope),
    ca       = factor(ca),
    thal     = factor(thal)
  )

cat("\nTarget distribution:\n")
print(table(heart_data$target))
cat("\nClass proportions:\n")
print(prop.table(table(heart_data$target)))

##############################################
# 3. Exploratory Data Analysis
##############################################

# Summary statistics
cat("\n--- Summary Statistics ---\n")
print(summary(heart_data))

# Age distribution by heart disease status
p1 <- ggplot(heart_data, aes(x = age, fill = target)) +
  geom_histogram(binwidth = 5, position = "dodge", alpha = 0.7) +
  labs(title = "Age Distribution by Heart Disease Status",
       x = "Age (years)", y = "Count", fill = "Heart Disease") +
  theme_minimal()
print(p1)

# Cholesterol vs max heart rate
p2 <- ggplot(heart_data, aes(x = chol, y = thalach, color = target)) +
  geom_point(alpha = 0.6) +
  labs(title = "Cholesterol vs Max Heart Rate",
       x = "Serum Cholesterol (mg/dl)",
       y = "Max Heart Rate Achieved",
       color = "Heart Disease") +
  theme_minimal()
print(p2)

# Chest pain type by disease status
p3 <- ggplot(heart_data, aes(x = cp, fill = target)) +
  geom_bar(position = "fill") +
  labs(title = "Chest Pain Type vs Heart Disease",
       x = "Chest Pain Type", y = "Proportion", fill = "Heart Disease") +
  theme_minimal()
print(p3)

##############################################
# 4. Train/Test Split (80/20)
##############################################

set.seed(42)
train_idx <- createDataPartition(heart_data$target, p = 0.80, list = FALSE)
train_set <- heart_data[ train_idx, ]
test_set  <- heart_data[-train_idx, ]

cat("\nTraining set rows:", nrow(train_set), "\n")
cat("Test set rows:", nrow(test_set), "\n")

# Cross-validation control
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

##############################################
# 5. Model 1: Logistic Regression
##############################################

set.seed(42)
lr_model <- train(target ~ .,
                  data = train_set,
                  method = "glm",
                  family = "binomial",
                  metric = "ROC",
                  trControl = ctrl)

lr_preds <- predict(lr_model, newdata = test_set)
lr_cm    <- confusionMatrix(lr_preds, test_set$target, positive = "Yes")

cat("\n========== Logistic Regression ==========\n")
print(lr_cm)

##############################################
# 6. Model 2: Random Forest
##############################################

set.seed(42)
rf_model <- train(target ~ .,
                  data = train_set,
                  method = "rf",
                  metric = "ROC",
                  tuneGrid = data.frame(mtry = c(2, 4, 6)),
                  trControl = ctrl)

rf_preds <- predict(rf_model, newdata = test_set)
rf_cm    <- confusionMatrix(rf_preds, test_set$target, positive = "Yes")

cat("\n========== Random Forest ==========\n")
print(rf_cm)

# Variable importance
cat("\nVariable Importance (Random Forest):\n")
print(varImp(rf_model))

##############################################
# 7. Model 3: Support Vector Machine (SVM)
##############################################

set.seed(42)
svm_model <- train(target ~ .,
                   data = train_set,
                   method = "svmRadial",
                   metric = "ROC",
                   preProcess = c("center", "scale"),
                   trControl = ctrl)

svm_preds <- predict(svm_model, newdata = test_set)
svm_cm    <- confusionMatrix(svm_preds, test_set$target, positive = "Yes")

cat("\n========== SVM ==========\n")
print(svm_cm)

##############################################
# 8. Model Comparison
##############################################

results_df <- data.frame(
  Model       = c("Logistic Regression", "Random Forest", "SVM"),
  Accuracy    = c(
    lr_cm$overall["Accuracy"],
    rf_cm$overall["Accuracy"],
    svm_cm$overall["Accuracy"]
  ),
  Sensitivity = c(
    lr_cm$byClass["Sensitivity"],
    rf_cm$byClass["Sensitivity"],
    svm_cm$byClass["Sensitivity"]
  ),
  Specificity = c(
    lr_cm$byClass["Specificity"],
    rf_cm$byClass["Specificity"],
    svm_cm$byClass["Specificity"]
  )
)

cat("\n========== Model Comparison ==========\n")
print(results_df)

# Bar plot comparing accuracies
p4 <- ggplot(results_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.3, size = 4) +
  labs(title = "Model Accuracy Comparison",
       x = "Model", y = "Test Accuracy") +
  theme_minimal() +
  theme(legend.position = "none") +
  ylim(0, 1)
print(p4)

##############################################
# 9. ROC Curves
##############################################

# Get probability predictions
lr_prob  <- predict(lr_model,  newdata = test_set, type = "prob")[,"Yes"]
rf_prob  <- predict(rf_model,  newdata = test_set, type = "prob")[,"Yes"]
svm_prob <- predict(svm_model, newdata = test_set, type = "prob")[,"Yes"]

true_labels <- ifelse(test_set$target == "Yes", 1, 0)

roc_lr  <- roc(true_labels, lr_prob)
roc_rf  <- roc(true_labels, rf_prob)
roc_svm <- roc(true_labels, svm_prob)

plot(roc_rf,  col = "#E74C3C", lwd = 2,
     main = "ROC Curves - Heart Disease Models")
lines(roc_lr,  col = "#3498DB", lwd = 2)
lines(roc_svm, col = "#2ECC71", lwd = 2)
legend("bottomright",
       legend = c(
         paste0("Random Forest  (AUC = ", round(auc(roc_rf),  3), ")"),
         paste0("Logistic Reg.  (AUC = ", round(auc(roc_lr),  3), ")"),
         paste0("SVM            (AUC = ", round(auc(roc_svm), 3), ")")
       ),
       col = c("#E74C3C", "#3498DB", "#2ECC71"),
       lwd = 2)

##############################################
# 10. Final Results
##############################################

best_model_idx <- which.max(results_df$Accuracy)
cat("\n===== FINAL RESULTS =====\n")
cat("Best Model:", results_df$Model[best_model_idx], "\n")
cat("Best Accuracy:", round(results_df$Accuracy[best_model_idx], 4), "\n")
cat("Best Sensitivity:", round(results_df$Sensitivity[best_model_idx], 4), "\n")
cat("Best Specificity:", round(results_df$Specificity[best_model_idx], 4), "\n")
cat("Random Forest AUC:", round(auc(roc_rf), 4), "\n")
