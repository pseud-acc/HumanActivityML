# Compute predictions for Holdout sample

#load packages
library(ggplot2);library(dplyr);library(reshape2);library(caret)
library(xgboost);library(ggpubr);library(data.table)

#load trained model data
load("model_data/final_model_data.RData")

#apply pca model to holdout data
holdoutpca <- predict(preproc, holdout[,-53])

#Convert input data into Dmatrices for XGB model
X_holdout = xgb.DMatrix(as.matrix(holdoutpca))

#generate base learner predictions on the holdout set
predsh_xgb <- predict(object = mod_xgb, newdata = X_holdout)
predsh_rf <- predict(object = mod_rf, newdata = holdoutpca)
predsh_knn <- predict(object = mod_knn, newdata = holdoutpca)

#combine base learner holdout set predictions
predsh_df <- data.frame(preds_xgb = predsh_xgb, 
                        preds_rf = predsh_rf, 
                        preds_knn = predsh_knn)

#Feed ensemble learner with base learner predictions
predsh_stack <- data.frame(pred = predict(mod_stack,predsh_df))
predsh_stack

#Check probabilities
predict(mod_stack,predsh_df,type="prob")
