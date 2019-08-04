# disease-prediction
library(e1071)
library(caret)
library(ROCR)
library(party)
library(randomForest)
library(neuralnet)
library(rpart)
library(psych)
library(devtools)
library(gdata)


# Read Data
data <- read.csv("diabetes.csv", header = TRUE)
str(data)
data$Outcome <- as.factor(data$Outcome)
table(data$Outcome)

# Data Partition
set.seed(123)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]

#random forest
rf <- randomForest(Outcome~., data=train,
                   ntree = 300,
                   mtry = 2,
                   importance = TRUE,
                   proximity = TRUE)

p_rf <- predict(rf, test)
tab2 <- table(p_rf, test$Outcome)
CM_rf <- confusionMatrix(p_rf, test$Outcome)
CM_rf
MCE_rf <- 1-sum(diag(tab2))/sum(tab2)

#MODEL PERFORMANCE
p_rf <- predict(rf, test, type = 'prob')
performance_rf <- prediction(p_rf[,2], test$Outcome)
ROC_rf <- performance(performance_rf,'tpr','fpr')
plot(ROC_rf,colorize = TRUE, main = "Random Forest")
abline(a=0,b=1)
auc <- performance(performance_rf,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)*100

legend(.45,.4,paste(auc,"%"),title = "AUC", cex = 1.5)




# Multi-dimensional Scaling Plot of the dataset
MDSplot(rf, train$Outcome, main = "Diabetics Data")


#Naive Bayes
set.seed(222)
NB <- naiveBayes(Outcome~., data = train)
NB

# Prediction & Confusion Matrix - test data
p_NB <- predict(NB, test)
tab2 <- table(p_NB, test$Outcome)
CM_NB <- confusionMatrix(p_NB, test$Outcome)
MCE_NB <- 1-sum(diag(tab2))/sum(tab2)

#performance evaluation
p_NB <- predict(NB, test, type="raw")
performance_NB <- prediction(p_NB[,2], test$Outcome)
ROC_NB <- performance(performance_NB,'tpr','fpr')
plot(ROC_NB, colorize = TRUE, main = "Naive Bayes")
abline(a=0,b=1)
auc <- performance(performance_NB,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)*100

legend(.45,.4,paste(auc,"%"),title = "AUC", cex = 1.5)


###############################
# Support Vector Machine
###############################
set.seed(222)
mysvm <- svm(Outcome~., train)
summary(mysvm)
attributes(mysvm)

# Prediction & Confusion Matrix - test data
p_svm <- predict(mysvm, test)
tab2 <- table(p_svm, test$Outcome)
CM_svm <- confusionMatrix(p_svm, test$Outcome)
CM_svm
CM_svm$byClass
MCE_svm <- 1-sum(diag(tab2))/sum(tab2)


#performance evaluation
#svm
p_svm <- predict(mysvm, test, decision.values = TRUE)
p_svm.prob <-  attr(p_svm,"decision.values")
performance_svm <- prediction(p_svm.prob, test$Outcome)
ROC_svm <- performance(performance_svm,'tpr','fpr')
plot(ROC_svm,  colorize = TRUE, main = "Support Vector Machine" )
abline(a=0,b=1)
auc <- performance(performance_svm,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)*100

legend(.45,.4,paste(auc,"%"),title = "AUC", cex = 1.5)




###################
###################
#neural network####
###################
###################

data <- read.csv("diabetes.csv", header = TRUE)
str(data)


# Min-Max Normalization
data$Pregnancies <- (data$Pregnancies - min(data$Pregnancies))/(max(data$Pregnancies) - min(data$Pregnancies))
data$Glucose <- (data$Glucose - min(data$Glucose))/(max(data$Glucose) - min(data$Glucose))
data$BloodPressure <- (data$BloodPressure - min(data$BloodPressure))/(max(data$BloodPressure)-min(data$BloodPressure))
data$SkinThickness <- (data$SkinThickness - min(data$SkinThickness))/(max(data$SkinThickness) - min(data$SkinThickness))
data$Insulin <- (data$Insulin - min(data$Insulin))/(max(data$Insulin) - min(data$Insulin))
data$BMI <- (data$BMI - min(data$BMI))/(max(data$BMI)-min(data$BMI))
data$DiabetesPedigreeFunction <- (data$DiabetesPedigreeFunction - min(data$DiabetesPedigreeFunction))/(max(data$DiabetesPedigreeFunction) - min(data$DiabetesPedigreeFunction))
data$Age <- (data$Age - min(data$Age))/(max(data$Age) - min(data$Age))
data$Outcome <- (data$Outcome - min(data$Outcome))/(max(data$Outcome)-min(data$Outcome))

# Data Partition
set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
training <- data[ind==1,]
testing <- data[ind==2,]

# Neural Networks
library(neuralnet)
library(devtools)
library(nnet)
set.seed(333)


nn <- neuralnet(Outcome~Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age,
                data = training,
                hidden = c(2),
                err.fct = "ce",
                stepmax = 1000000000,
                threshold = 0.01,
                linear.output = FALSE)

plot(nn)


# Confusion Matrix & Misclassification Error - testing data
output <- compute(nn, testing[,-9])
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab_nn <- t(table(pred2, testing$Outcome))

CM_nn <- confusionMatrix(tab_nn)
CM_nn
MCE_nn <- 1-sum(diag(tab_nn))/sum(tab_nn)
MCE_nn

#performance evaluation
output <- compute(nn, testing[,-9])
p_ann <- output$net.result
performance_ann <- prediction(p_ann, testing$Outcome)
ROC_ann <- performance(performance_ann,'tpr','fpr')
plot(ROC_ann,  colorize = TRUE ,main = "Neural Network")
abline(a=0,b=1)
auc <- performance(performance_ann,"auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)*100

legend(.45,.4,paste(auc,"%"),title = "AUC", cex = 1.5)










#accuracy graph


Accuracy <- matrix(c(CM_NB$overall['Accuracy'],
                     CM_svm$overall['Accuracy'],
                     CM_rf$overall['Accuracy'],
                     CM_nn$overall['Accuracy']
                     ),
                   dimnames = list(c("NB","SVM","rf","ANN")
                   ))
Accuracy <- round(Accuracy*100,2)
Accuracy

barplot(Accuracy, beside = TRUE, col = c("Violet","skyblue","red","blue"), 
        xlab = "Models", ylab = "Model Accuracy in parcentage", ylim = c(0,100), xlim = c(1,13),
        main = "Accuracy")

legend(x=5.5, y= 40, legend = c("NB","SVM","rf","ANN"), fill = c("Violet","skyblue","red","blue"))





#misclassification error
MCE <- matrix(c(MCE_NB,MCE_svm,MCE_rf,MCE_nn),
              dimnames = list(c("NB","SVM","rf","ANN")
              ))

MCE <- round(MCE*100,2)

barplot(MCE, beside = TRUE, col = c("Violet","skyblue","red","blue"), 
        xlab = "Models", ylab = "Missclassification Error in parcentage",ylim = c(0,40), xlim = c(1,10),
        main = "MCE comparison")
legend(x=5.5, y= 40, legend = c("NB","SVM","rf","ANN"), fill = c("Violet","skyblue","red","blue"))



tpr_fpr <- matrix(c(CM_NB$byClass['Sensitivity'], 
                    CM_NB$byClass['Specificity'],
                    CM_svm$byClass['Sensitivity'], 
                    CM_svm$byClass['Specificity'],
                    CM_rf$byClass['Sensitivity'], 
                    CM_rf$byClass['Specificity'],
                    CM_nn$byClass['Sensitivity'], 
                    CM_nn$byClass['Specificity']),
                  nrow = 2,
                  dimnames = list(c("Sensitivity","Specificity"),
                                  c("NB","SVM","RF", "ANN")
                  ))
tpr_fpr <- round(tpr_fpr*100,2)

barplot(tpr_fpr, beside = TRUE, col = c("blueviolet","aquamarine2"), 
        xlab = "Models", ylab = "parformance in parcentage", ylim = c(0,100),
        main = "Performance")
legend(x=01, y= 98, legend = c("Sensitivity","Specificity"), fill = c("blueviolet","aquamarine2"))


