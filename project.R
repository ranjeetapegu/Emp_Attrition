# Employee_attrition rate
library(caret)
setwd("~/Documents/HARVARD Studies/Fundamental of DataScience/project")

Emp.Attrition = read.csv("Employee-Attrition.csv")
head(Emp.Attrition,10)
summary(Emp.Attrition)
library(mosaic)

tally(~Attrition | MaritalStatus, format = "proportion", data = Emp.Attrition, margins=TRUE)

names(Emp.Attrition)
Emp.Attrition = Emp.Attrition[,-c(22,27)]
library(ggplot2)

Emp.Attrition[1:10, c(1,2,4,13,19,20,23)]
attach(Emp.Attrition)
ggplot(data =Emp.Attrition, aes(x=Attrition, y= Age,col= Attrition)) + 
    geom_boxplot(aes(fill=Attrition))

cat_cols = c("Age","DistanceFromHome","HourlyRate","MonthlyIncome","MonthlyRate"
             ,"TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany",
             "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager")

 c("MonthlyIncome","OverTime","NumCompaniesWorked",
   "DailyRate","MonthlyRate","DailyRate","DistanceFromHome"))
    
    
    
ggplot(data =Emp.Attrition, aes(DistanceFromHome,fill=Attrition)) +
    geom_histogram() + facet_grid(~Attrition)

bargraph( ~ Department+Attrition, data = Emp.Attrition)
tally(~Class, format = "count", data = Titanic)
mosaicplot(~ Department+Attrition, data = Emp.Attrition, color=TRUE)
    
    
    
ggplot(data =Emp.Attrition, 
          aes( Education,Attrition,fill= Attrition)) +
          geom_bar(stat = "identity") 

ggplot(data =Emp.Attrition, 
       aes( WorkLifeBalance,Attrition,fill= Attrition)) +
    geom_bar(stat = "identity") 


MonthlyRate    NumCompaniesWorked Over18   OverTime   

dim(Emp.Attrition)
pairs(Emp.Attrition[,c('StockOptionLevel',
                       'TotalWorkingYears',
                       'TrainingTimesLastYear',
                       'WorkLifeBalance' ,
                       'YearsAtCompany' ,
                       'YearsInCurrentRole' ,
                       'YearsSinceLastPromotion' ,
                       'YearsWithCurrManager')])

plot(x=WorkLifeBalance,y = MonthlyIncome,col = Attrition)

cor(Emp.Attrition)
#data cleansing
Emp.Attrition = Emp.Attrition [, -c(22,27)]
 
 # Classification Problem
# Logistic Regression
library(MASS)
dtrain = createDataPartition(Emp.Attrition$Attrition, p =0.75, list= FALSE)
training = Emp.Attrition[dtrain,]
test= Emp.Attrition[-dtrain,]
Log_m = glm(Attrition~ Age + BusinessTravel+ DailyRate
            +Department + DistanceFromHome + Education +
                EducationField + EnvironmentSatisfaction +
                Gender +HourlyRate + JobInvolvement + JobLevel+
                JobRole +   JobSatisfaction + MaritalStatus+ MonthlyIncome +
                MonthlyRate+NumCompaniesWorked + OverTime  + PercentSalaryHike +
                PerformanceRating+ RelationshipSatisfaction +
                StockOptionLevel + TotalWorkingYears+
                TrainingTimesLastYear+ WorkLifeBalance + YearsAtCompany,
            family="binomial",
            data = training)
summary(Log_m)
pred_link = predict(Log_m,newdata= test,type="link")
pred = predict(Log_m,newdata= test,type="response")
glm.pred = rep("No",dim(test)[1])
glm.pred[pred >0.5]="Yes"
confusionMatrix(glm.pred, test$Attrition)
score_data <- data.frame(link=pred_link, 
                         response=pred,
                         Attrition=test$Attrition,
                         stringsAsFactors=FALSE)

Log_m_1 = glm(Attrition ~ BusinessTravel + DistanceFromHome + EnvironmentSatisfaction + 
        JobInvolvement +JobSatisfaction + NumCompaniesWorked  + OverTime + 
        TrainingTimesLastYear  + TrainingTimesLastYear,family="binomial",data = training )
summary(Log_m_1)
pred = predict(Log_m_1,newdata= test,type="response")
glm.pred = rep("No",dim(test)[1])
glm.pred[pred >0.5]="Yes"
confusionMatrix(glm.pred, test$Attrition)
library(ROCR)
pred1 = prediction(predict(Log_m_1,newdata= test),test$Attrition)
perf1 = performance(pred1,"tpr","fpr")
plot(perf1)
#LDA or QDA
lda.fit = lda(Attrition~ Age + BusinessTravel+ DailyRate
              +Department + DistanceFromHome + Education +
                  EducationField + EnvironmentSatisfaction +
                  Gender +HourlyRate + JobInvolvement + JobLevel+
                  JobRole +   JobSatisfaction + MaritalStatus+ MonthlyIncome +
                  MonthlyRate+NumCompaniesWorked + OverTime  + PercentSalaryHike +
                  PerformanceRating+ RelationshipSatisfaction +
                  StockOptionLevel + TotalWorkingYears+
                  TrainingTimesLastYear+ WorkLifeBalance + YearsAtCompany,
              data = training)
plot(lda.fit)
pred = predict(lda.fit,test)$class
confusionMatrix(pred, test$Attrition)

# QDA 
qda.fit = qda(Attrition~ Age + BusinessTravel+ DailyRate 
              +Department + DistanceFromHome + Education 
              + EducationField + EnvironmentSatisfaction +
              Gender +HourlyRate + JobInvolvement + JobLevel
                + JobSatisfaction + MaritalStatus+ MonthlyIncome
                + MonthlyRate+NumCompaniesWorked + OverTime  + PercentSalaryHike 
                + PerformanceRating+ RelationshipSatisfaction +
                StockOptionLevel + TotalWorkingYears+
                TrainingTimesLastYear+ WorkLifeBalance + YearsAtCompany,
              #+ JobRole,
              data = training)

pred = predict(qda.fit,test)$class
confusionMatrix(pred, test$Attrition)

## KNN model prediction
set.seed(1)
library(FNN)
train= training[,c(1,2,4,6,7,11,14,14,15,19,20,21,24,25,26,28,29,30,31,32,33,34,35)]
testing = test[,c(1,2,4,6,7,11,14,14,15,19,20,21,24,25,26,28,29,30,31,32,33,34,35)]

set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(Attrition ~ ., data = train, 
                method = "knn", 
                trControl = ctrl, 
                preProcess = c("center","scale"), tuneLength = 20)
plot(knnFit)
knn.pred = predict(knnFit,newdata =testing )
confusionMatrix(knn.pred,Y_test)
# Random forest 

trctrl = trainControl(method = "cv",number =10,repeats =3)
set.seed(123)

rf_mdl = train(Attrition~ Age + BusinessTravel+ DailyRate
               +Department + DistanceFromHome + Education +
                   EducationField + EnvironmentSatisfaction +
                   Gender +HourlyRate + JobInvolvement + JobLevel+
                   JobRole +   JobSatisfaction + MaritalStatus+ MonthlyIncome +
                   MonthlyRate+NumCompaniesWorked + OverTime  + PercentSalaryHike +
                   PerformanceRating+ RelationshipSatisfaction +
                   StockOptionLevel + TotalWorkingYears+
                   TrainingTimesLastYear+ WorkLifeBalance + YearsAtCompany,
                data = training, method ="rf",prox = FALSE, trControl = trctrl)
print(rf_mdl, digits = 4)
pred <- predict(rf_mdl, newdata = test)
confusionMatrix(pred, test$Attrition)
# lasso

x = model.matrix(Attrition~ Age + BusinessTravel+ DailyRate
                 +Department + DistanceFromHome + Education +
                     EducationField + EnvironmentSatisfaction +
                     Gender +HourlyRate + JobInvolvement + JobLevel+
                     JobRole +   JobSatisfaction + MaritalStatus+ MonthlyIncome +
                     MonthlyRate+NumCompaniesWorked + OverTime  + PercentSalaryHike +
                     PerformanceRating+ RelationshipSatisfaction +
                     StockOptionLevel + TotalWorkingYears+
                     TrainingTimesLastYear+ WorkLifeBalance + YearsAtCompany, 
                 data= Emp.Attrition)[ ,- 1]

y =Emp.Attrition[dtrain,"Attrition"]
y= ifelse(y=="No",0,1)
Train <- x[dtrain,]
test= x[-dtrain,]

head(x)
grid=10^seq(10,-3,length=100) # lambda range
library(glmnet)
lasso.Mod <- glmnet(Train,y,alpha=1, lambda =grid)
#ploting coeff vs. L1 Norm
plot(lasso.Mod,"norm",ylim=c(0,0.4))
cvLassoRes <- cv.glmnet(Train,y,alpha=1) 
predict(lasso.Mod,type="coefficients",s=cvLassoRes$lambda.1se)
lassoTestPred <- predict(lasso.Mod,newx=test,s=cvLassoRes$lambda.1se)
range(lassoTestPred)
pred = as.factor(ifelse(lassoTestPred >0.5,"Yes","No"))
confusionMatrix(pred, Emp.Attrition[-dtrain,"Attrition"])

###
library(e1071)
dtrain = createDataPartition(Emp.Attrition$Attrition, p =0.75, list= FALSE)
training = Emp.Attrition[dtrain,]
test= Emp.Attrition[-dtrain,]
svm.fit.l = svm(Attrition~ Age + BusinessTravel+ DailyRate
                +Department + DistanceFromHome + Education +
                    EducationField + EnvironmentSatisfaction +
                    Gender +HourlyRate + JobInvolvement + JobLevel+
                    JobRole +   JobSatisfaction + MaritalStatus+ MonthlyIncome +
                    MonthlyRate+NumCompaniesWorked + OverTime  + PercentSalaryHike +
                    PerformanceRating+ RelationshipSatisfaction +
                    StockOptionLevel + TotalWorkingYears+
                    TrainingTimesLastYear+ WorkLifeBalance + YearsAtCompany,
                data = training, kernel="linear",cost=1,scale= FALSE)
plot(svm.fit.l,training)
pred = predict(svm.fit.l, newdata = test)
confusionMatrix(pred, test$Attrition)


library(caret)
trctrl = trainControl(method = "repeatedcv",number =10,repeats =3)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))

set.seed(3233)
svm_l_cv = train(Attrition~ Age + BusinessTravel+ DailyRate
                 +Department + DistanceFromHome + Education +
                     EducationField + EnvironmentSatisfaction +
                     Gender +HourlyRate + JobInvolvement + JobLevel+
                     JobRole +   JobSatisfaction + MaritalStatus+ MonthlyIncome +
                     MonthlyRate+NumCompaniesWorked + OverTime  + PercentSalaryHike +
                     PerformanceRating+ RelationshipSatisfaction +
                     StockOptionLevel + TotalWorkingYears+
                     TrainingTimesLastYear+ WorkLifeBalance + YearsAtCompany,
                 data = training,
                 method ="svmLinear",
                 trainControl= trctrl ,
                 tuneGrid =grid,
                 tuneLength =10)
plot(svm_l_cv)
pred = predict(svm_l_cv , newdata = test)
confusionMatrix(pred, test$Attrition)

trctrl = trainControl(method = "cv",number =10,repeats =3)
grid <- expand.grid(degree= 3,
                    C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
                    scale= TRUE)

set.seed(3233)
svm_l_cv = train(Attrition~ Age + BusinessTravel+ DailyRate
                 +Department + DistanceFromHome + Education +
                     EducationField + EnvironmentSatisfaction +
                     Gender +HourlyRate + JobInvolvement + JobLevel+
                     JobRole +   JobSatisfaction + MaritalStatus+ MonthlyIncome +
                     MonthlyRate+NumCompaniesWorked + OverTime  + PercentSalaryHike +
                     PerformanceRating+ RelationshipSatisfaction +
                     StockOptionLevel + TotalWorkingYears+
                     TrainingTimesLastYear+ WorkLifeBalance + YearsAtCompany,
                 data = training,
                 method ="svmPoly",
                 preProcess = c("center", "scale"),
                 trainControl= trctrl ,
                 tuneGrid =grid,
                 tuneLength =10)

plot(svm_l_cv)
pred = predict(svm_l_cv , newdata = test)
confusionMatrix(pred, test$Attrition)