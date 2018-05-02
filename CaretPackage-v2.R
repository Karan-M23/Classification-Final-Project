#################################################
#                    Packages                   #
#################################################

library(caret)
library(caTools) 
library(doParallel)
library(kernlab)
library(klaR)
library(MASS)
library(modelr)
library(pROC)
library(rpart)
library(randomForest)

#################################################
#                 Data Importing                #
#################################################
## Read Data
#setwd("/Karan_M/Documents/STAT 441")
#data <- read.csv('default of credit card clients.csv')
data <- read.csv("Documents/STAT 441/default of credit card clients.csv")
#colnames(data)[25] <- 'default.payment.next.month'
#################################################
#                 Data Processing               #
#################################################
## Defining factors that will be characteristics
factor_vars <- c('SEX','EDUCATION','MARRIAGE','default.payment.next.month','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6')
data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))
levels(data$default.payment.next.month) <- list(no="0", yes="1") 
# Binning ages
data <- transform(data, MARRIAGE_BIN = ifelse(MARRIAGE==0, 3, MARRIAGE))  # marriage = 0 becomes 3
data <- transform(data, EDUCATION_BIN = ifelse(EDUCATION==5 | EDUCATION==6, 4, EDUCATION))  # education = 5 or 6 becomes 4
age_int <- seq(from=20, to=80, by=5)
#data$AGE_INT <- cut(data$AGE,age_int) # for reference, if you wanted to see which age interval AGE_BIN fell into you can uncomment this
data$AGE_BIN <- cut(data$AGE,age_int,labels=FALSE) # age banded into intervals of 5 years
#data_mod <- subset(data, select=-c(AGE_INT,ID)) #can use this line to drop certain columns in select=-c(.) if the dataset is too large 

# Reduced Model Analysis
data_reduced <- step(glm(default.payment.next.month ~ ., data = data, family = binomial),
                     direction = 'both')
data_reduced_col = c(all.vars(formula(data_reduced)[-2]), 'default.payment.next.month')
data_reduced_final = data[data_reduced_col]

#################################################
#                 Data Splitting                #
#################################################
# divide into train and test set
set.seed(5309)
# 80% Training 20% Testing
split = sample.split(data$default.payment.next.month, SplitRatio = 0.8)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

train_set_reduce = subset(data_reduced_final, split == TRUE)
test_set_reduce = subset(data_reduced_final, split == FALSE)

#################################################
#                 Data Tuning                   #
#################################################
# https://topepo.github.io/caret/model-training-and-tuning.html#control

# Tuning Control: 
# TwoClassSummary - Caret will find the best hyperparmeter in regards to sensitivity, specificity, and area under ROC curve.
# ClassProbs - Compute the class probability
# Goal is to find the best hyper parameter for each model before we train
tuner <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)


# Evaluation Control:
# We are using Repeated K-Fold CV, 10 folds repeated 5 times.
# Still use ClassProb and calculate TwoClassSummary
# Save Prediction to pass onto predict later
evaluater <- trainControl(method = "repeatedcv" , number = 10, repeats = 5, 
                          savePredictions = T, summaryFunction = twoClassSummary, 
                          classProbs = T)
grid <- expand.grid(fL=0, usekernel = TRUE, adjust=1)

# Tuning 
# Tune Length = 2. Caret will find optimal parameter and evaluate two of them
train(default.payment.next.month ~ ., 
        metric = "ROC", preProc = c("center", "scale"), 
        tuneLength = 3, trControl = tuner, data = train_set, method = "svmLinear")

train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = train_set_reduce, method = "adaboost")

# Alternatively, for multiple parameter models, use 
# grid <- expand.grid(size=c(5,10,20,50), k=c(1,2,3,4,5))
# tuneGrid = grid replaces tuneLength in the above code
 

#24000 samples
#27 predictor
#2 classes: 'no', 'yes' 

#Pre-processing: centered (86), scaled (86) 
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 24000, 24000, 24000, 24000, 24000, 24000, ... 
#Resampling results across tuning parameters:
#  
#  k   ROC        Sens       Spec     
#5  0.6793716  0.8767672  0.3747946
#7  0.6933661  0.8991700  0.3617133
#9  0.7027048  0.9136846  0.3544072
#11  0.7100527  0.9237415  0.3507521
#13  0.7153684  0.9304767  0.3470546##

#ROC was used to select the optimal model using the largest value.
#The final value used for the model was k = 13. KNN

#################################################
#                 Data Training                 #
#################################################
# Above code will train to find the best hyper parameter to use in the model
# Please follow the following convention model_METHODNAME

model_NB <- train(default.payment.next.month ~ ., data = train_set,
                   metric = "ROC", preProc = c("center", "scale"),
                   tuneGrid = data.frame(fL=0, usekernel = TRUE, adjust=1), trControl = evaluater, method = "nb")

saveRDS(model_NB, "model_NB.rds")

model_KNN_reduced <- train(default.payment.next.month ~ ., data = train_set_reduce,
                   metric = "ROC", preProc = c("center", "scale"),
                   tuneGrid = data.frame(k = #WHATEVER YOUR OPTIMAL PARAMETERS ARE
                   ), trControl = evaluater, method = "knn")

saveRDS(model, "model_KNN_reduced.rds")
 
#################################################
#                Data Prediction                #
#################################################
predict_NB <- predict(model_NB, newdata = test_set)
predict_KNN_reduce <- predict(model_KNN_reduced, newdata = test_data)

## Give me suggestions here as to how to demonstrate data. Confusion matrix? Help

                 