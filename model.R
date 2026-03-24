# siner
# Artifical Intelligence Model using Machine learning Algorithm to predict  the ICO ( Cryptocurrencies ) offerings for Fundraising teams and Startups
rm(list=ls())
icobench<- read.csv("Downloads/Machinelearningcourseworkright.csv")
View(icobench)
str(icobench)
icobench$success<-factor(icobench$success)
str(icobench$success)
sum(complete.cases(icobench)) 
sum(!complete.cases(icobench))
summary(icobench)
View(icobench)


install.packages("VIM")
library("VIM")
aggr(icobench,numbers=TRUE,prop=FALSE)
qicobench<-icobench[which(icobench$hasVideo==1&icobench$rating>=4.0),]
hist(icobench$priceUSD)
si_icobench<-icobench
si_icobench$priceUSD[is.na(si_icobench$priceUSD)]<-mean(si_icobench$priceUSD,na.rm = TRUE)
si_icobench
hist(icobench$teamSize)
si_icobench$teamSize[is.na(si_icobench$teamSize)]<-mean(si_icobench$teamSize,na.rm = TRUE)
 
si_icobench

str(icobench$success)
si_icobench$success[is.na(si_icobench$success)]<-mean(si_icobench$success,na.rm=TRUE)
View(si_icobench)

crowdfunding<-si_icobench
View(crowdfunding)
sum(complete.cases(si_icobench))
sum(!complete.cases(crowdfunding))
dup_crowfunding<-duplicated(crowdfunding$hasVideo)
x<-si_icobench
x<-x[-which(x$success%in%outliers),]
table(x$success,x$minInvestment)
table(data$success,data$hasReddit)
boxplot(x$priceUSD,plot = FALSE)$out # priceUSD,teamsize,coinnum,dirtubutedpercentage are the outliers
outlier<-boxplot(x$priceUSD,plot = FALSE)$out
length(x$priceUSD)
length(x$teamSize)
boxplot(x$priceUSD)
sum(complete.cases(x))
sum(!complete.cases(x))
outliers<-boxplot(x$teamSize,plot = FALSE)$out
y<-x
y<-y[-which(y$teamSize %in% outliers),]
length(y)
length(y$priceUSD)
out<-boxplot(y$coinNum,plot = FALSE)$out
z<-y
z<-z[-which(z$coinNum %in% out),]
length(z$teamSize)
sum(!complete.cases(z))
outl<-boxplot(z$distributedPercentage,plot = FALSE)$out
w<-z
w<-w[-which(w$distributedPercentage %in% outl),]
sum(complete.cases(w))
# KNN CLASSIFICATION TECHNIQUE
crowdfunding<-crowdfunding[-1]#Removing the id factor 
table(crowdfunding$success)
crowdfunding$success<-as.character(crowdfunding$)
View(crowdfunding)
round(prop.table(table(crowdfunding$success))*100 , digits = 1)
summary(crowdfunding[c("rating","priceUSD","teamSize")])
normalize <- function(x) {
  if(is.numeric(x)){
  x= ((x - min(x)) / (max(x) - min(x)))
  }
  return(x)
    }
crowdfunding_n<-as.data.frame(lapply(crowdfunding[2:10],normalize))
summary(crowdfunding_n$priceUSD)
crowdfunding_train<-crowdfunding_n[1:2075,]
crowdfunding_test<-crowdfunding_n[2076:2767,]
crowdfunding_train_labels<-crowdfunding[1:2075,1]
crowdfunding_test_labels<-crowdfunding[2076:2767,1]
crowdfunding_train_labels
install.packages("class")
library(class)

# We use knn() function to perform classification
# We split our data into training and test datasets, 
# each with exactly the same numeric features. 
# The labels for the training data are stored 
# in a separate factor vector. 
# The only remaining parameter is k, 
# which specifies the number of neighbors to include in the vote.
K=45
# training size =2666 so we try its square root 51 as the value of k first
#Using an odd number of K will reduce the chance of ending with a tie vote.
crowdfunding_test_pred<-knn(train =crowdfunding_train,test = crowdfunding_test, cl=crowdfunding_train_labels, k=K )
#error while specifying the crowdfunding_train_labels
install.packages("gmodels")
library(gmodels)
CrossTable(crowdfunding_test_labels,crowdfunding_test_pred,   prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actua
                   l'))
(542+3)/692

install.packages("Metrics")
library(Metrics)
mae(crowdfunding_test_pred,crowdfunding_test)
library(ROCR)
icobench_test_prob<-predict(crowdfunding_test_pred,crowdfunding_test,type = "raw")


options(scipen=999)
head(icobench_test_prob)
icobench_results<-data.frame(actual_type=crowdfunding_test_labels,
                              predict_type=crowdfunding_test_pred,
                             prob_yes=round(crowdfunding_test_prob[,1],5),
                             prob_no=round(crowdfunding_test_prob[,2],5))


head(icobench_results)
head(subset(icobench_results,actual_type!=predict_type))
CrossTable(icobench_results$actual_type,icobench_results$predict_type, dnn=c('actual', 'predict'),
           prop.chisq = FALSE, prop.t = FALSE, prop.r=FALSE)

install.packages("caret", dependencies = TRUE)
library(caret)
confusionMatrix(icobench_results$predict_type, icobench_results$actual_type, positive = "N")
library(ROCR)
ROC<-roc(crowdfunding_test_labels,as.numeric(crowdfunding_test_pred),measure = "tpr",x.measure = "fp")
plot(ROC)
auc(ROC)
pred_object<-prediction(icobench_results$prob_yes,icobench_results$actual_type)
roc_NB<-performance(pred_object,measure = "tpr",x.measure = "fpr")
plot(roc_NB, main="ROC of success of companies for crowdfunding ",col="blue",lwd=2)
abline(a = 0, b = 1, lwd = 2, lty = 2)
auc_object_NB <- performance(pred_object, measure = "auc")
auc_NB <- auc_object_NB@y.values[[1]]


# Naive Bayes Classifers
rm(list=ls())
icobench<-read.csv("Downloads/MLNB.csv")
View(icobench)
icobench<-icobench[-1]
View(icobench )
str(icobench)
icobench$success<-factor(icobench$success)
str(icobench$success) 
table(icobench$success)
install.packages("tm")
library(tm)
icobench_corpus<-VCorpus(VectorSource(icobench$brandSlogan))
print(icobench_corpus)
inspect(icobench_corpus[1:2])
as.character(icobench_corpus[1])
lapply(icobench_corpus[1:5], as.character)
# clean up the corpus using tm_map()
icobench_corpus_clean<-tm_map(icobench_corpus,content_transformer(tolower))
# show the difference between sms_corpus and corpus_clean
as.character(icobench_corpus[[1]])
as.character(icobench_corpus_clean[[1]])

icobench_corpus_clean<-tm_map(icobench_corpus_clean,removeNumbers)
icobench_corpus_clean <- tm_map(icobench_corpus_clean, removeWords, stopwords()) # remove stop words
icobench_corpus_clean <- tm_map(icobench_corpus_clean, removePunctuation) # remove punctuat
install.packages("SnowballC")
library(SnowballC)
icobench_corpus_clean<-tm_map(icobench_corpus_clean,stemDocument)
icobench_corpus_clean<-tm_map(icobench_corpus_clean,stripWhitespace)
lapply(icobench_corpus[1:3],as.character)
lapply(icobench_corpus_clean[1:3],as.character)
icobench_dtm <- DocumentTermMatrix(icobench_corpus_clean)
dim(icobench_dtm)
inspect(icobench_dtm[1:5, 1:10])
set.seed(123)

ratio=0.85
p_index=round(nrow(icobench_dtm)*ratio)
icobench_dtm_train<-icobench_dtm[1:p_index,]
icobench_dtm_test<-icobench_dtm[(p_index +1):nrow(icobench_dtm),]
icobench_train_labels<-icobench[1:p_index,]$success
icobench_test_labels<-icobench[(p_index +1):nrow(icobench_dtm),]$success
prop.table(table(icobench_train_labels))
prop.table(table(icobench_test_labels))
findFreqTerms(icobench_dtm,10)
icobench_Freq_words<-findFreqTerms(icobench_dtm,10)
str(icobench_Freq_words)
icobench_dtm_Freq_train<-icobench_dtm_train[,icobench_Freq_words]
icobench_dtm_Freq_test<-icobench_dtm_test[,icobench_Freq_words]
# naive bayes clssifier is trained on categorical data
# convert counts to categorical variable
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
# apply() convert_counts() to columns of train/test data
# MARGIN=2 means apply function on column; MARGIN=1 means row
# search lapply() we used in last session and compare them
icobench_train<-apply(icobench_dtm_Freq_train, MARGIN = 2,convert_counts)
icobench_test<-apply(icobench_dtm_Freq_test, MARGIN = 2,convert_counts)
install.packages("e1071")
library(e1071)
icobench_classifier<-naiveBayes(icobench_train,icobench_train_labels)
icobench_test_pred<-predict(icobench_classifier,icobench_test)
library(gmodels)
CrossTable(icobench_test_pred, icobench_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
confusionMatrix(icobench_test_pred,icobench_test_labels,positive = TRUE,dnn = c("predicted","actual"))
(278+18)/415

## Step 5: Improving model performance ---
icobench_classifier_laplace<-naiveBayes(icobench_train,icobench_train_labels,laplace =1)
icobench_test_pred_laplace<-predict(icobench_classifier_laplace,icobench_test)
CrossTable(icobench_test_pred_laplace, icobench_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
#EVALUATION
icobench_test_prob<-predict(icobench_classifier,icobench_test,type = "raw")


options(scipen=999)
head(icobench_test_prob)
icobench_results<-data.frame(actual_type=icobench_test_labels,
                             predict_type=as.numeric(icobench_test_pred),
                             prob_yes=round(icobench_test_prob[,1],10),
                             prob_no=round(icobench_test_prob[,2],10))


head(icobench_results)
head(subset(icobench_results,actual_type!=predict_type))
CrossTable(icobench_results$actual_type,icobench_results$predict_type, dnn=c('actual', 'predict'),
           prop.chisq = FALSE, prop.t = FALSE, prop.r=FALSE)

(278+18)/415
install.packages("caret", dependencies = TRUE)

library(caret)
confusionMatrix(icobench_results$predict_type, icobench_results$actual_type, positive = "N")

library(pROC)
plot(roc((as.numeric(icobench_test_pred),icobench_test_label,direction="<",col="yellow",lwd = 3 ,main="ROC cCurve")))
ROCurve<-roc(icobench_test_labels , as.numeric(icobench_test_pred,measure = "tpr",x.measure = "fp"))
plot(ROCurve)
auc(ROCurve)
pred_object<-prediction(icobench_results$predict_type,icobench_results$actual_type)
roc_NB<-performance(pred_object,measure = "tpr",x.measure = "fpr")
plot(roc_NB, main="ROC of success of companies for crowdfunding ",col="blue",lwd=2)

abline(a = 0, b = 1, lwd = 2, lty = 2)
auc_object_NB <- performance(pred_object, measure = "auc")
auc_NB <- auc_object_NB@y.values[[1]]
auc_NB



#Decision Tree Analysis

rm(list=ls())
icobench<- read.csv("Downloads/MLDT.csv")
View(icobench)
str(icobench)
icobench$success<-factor(icobench$success)

str(icobench$success)
sum(complete.cases(icobench)) 
sum(!complete.cases(icobench))
summary(icobench)
install.packages("VIM")
library("VIM")
aggr(icobench,numbers=TRUE,prop=FALSE)
qicobench<-icobench[which(icobench$hasVideo==1&icobench$rating>=4.0),]
hist(icobench$priceUSD)
si_icobench<-icobench
si_icobench$priceUSD[is.na(si_icobench$priceUSD)]<-mean(si_icobench$priceUSD,na.rm = TRUE)
si_icobench
hist(icobench$teamSize)
si_icobench$teamSize[is.na(si_icobench$teamSize)]<-mean(si_icobench$teamSize,na.rm = TRUE)
sum(complete.cases(si_icobench))
ICO<-si_icobench
view(ICO)

table(ICO$startDate)
table(ICO$endDate)
summary(ICO$platform)
ICO$hasGithub<-factor(ICO$hasGithub)


smp_size<-floor(0.75*nrow(ICO))
sample(11,6)
sample(19,5)
set.seed(12345)
sample(10,5)
train_ico<-sample(nrow(ICO),smp_size)
crowdfunding_train<-ICO[train_ico,]
crowdfunding_test<-ICO[-train_ico,]
#--------------------------------------------------
#2. Training a model on the data: basic 
#--------------------------------------------------
#We will use the C5.0 algorithm in the 
#C50 package for training our decision tree model.ICO
ICO$hasGithub<-factor(ICO$hasGithub)
install.packages("C50")
library(C50)
library(tidyverse)
icobench_model<-C5.0(crowdfunding_train[6] ,crowdfunding_train$success)
icobench_model

summary(icobench_model)
crowdfunding_pred<-predict(icobench_model,crowdfunding_test)
install.packages("Metrics")
library('Metrics')
mae(crowdfunding_pred,crowdfunding_test$success)


#This creates a vector of predicted class values, 
#which we can compare to the actual class values 
#using the CrossTable() function in the gmodels package.
#Setting the prop.c and prop.r parameters to FALSE 
#removes the column and row percentages from the table. 
#The remaining percentage (prop.t) 
#indicates the proportion of records in the cell out of the total number of records.
library(gmodels)
CrossTable(crowdfunding_pred,crowdfunding_test$success,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted default', 'actual'))
confusionMatrix(crowdfunding_pred,crowdfunding_test$success,positive="Y")

crowdfunding_boost<-C5.0(select(crowdfunding_train[1]) ,crowdfunding_train$success,trials = 10)
crowdfunding_boost
summary(crowdfunding_boost)

crowdfunding_boost_pred<-predict(crowdfunding_boost,crowdfunding_test)
CrossTable(crowdfunding_boost_pred,crowdfunding_test$success,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted default', 'actual'))

# Regression Anlysis
rm(list=ls())
icobench<-read.csv("Downloads/Machinelearningcourseworkright.csv")
str(icobench)
icobench<-icobench[-1]
view(icobench)
icobench$success<-factor(icobench$success)

str(icobench$success)
smp_size<-floor(0.75*nrow(icobench))
set.seed(987)
train_ind<-sample(nrow(icobench),smp_size)

icobench_train<-icobench[train_ind,]

icobench_test<-icobench[-train_ind,]
library(rpart)
m.rpart<-rpart(success~.,data = icobench)
summary(m.rpart)
install.packages('rpart.plot')
library(rpart.plot)
rpart.plot(m.rpart, digits =2)
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)
p.rpart<-predict(m.rpart,icobench_test)
library(gmodels)
CrossTable(p.rpart,icobench_test$success,prop.chisq = FALSE, prop.c = FALSE, prop.r=FALSE ,dnn=c("predicted","actual"))


install.packages("Metrics")
library(Metrics)
mae(p.rpart,icobench_test$success)
rmse(p.rpart,icobench_test$success)
baseline<-mean(icobench_train$success)
mae(baseline,icobench_test$success)
rmse(baseline,icobench_test$success)
library(Cubist)
library(tidyverse)
m.cubist<-cubist(x=select(icobench_train,-success),y=icobench_train$success)
summary(m.cubist)
p.cubist<-predict(m.cubist,select(icobench_test,-success))
mae(icobench_test$success,p.cubist)
rmse(icobench_test$success,p.cubist)

#random Forest Clasifiers
install.packages("randomForest")
library(randomForest)
m.rf<-randomForest(icobench_test,y=icobench_train[1], ntree = 20)
p.rf<-predict(m.rf,icobench[train_ind==2,])
