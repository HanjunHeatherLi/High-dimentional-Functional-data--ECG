rm(list=ls())
set.seed(123)

#library
library(splines)
library(randomForest)
library(fda)

#read data
train_data = read.csv('ECG200TRAIN', header = FALSE)
#train_matrix = as.matrix(train_data,rownames = FALSE)
x_train = train_data[,-1]
train_label = as.factor(train_data[,1]) #1st column value to factors
train_label
test_data = read.csv('ECG200TEST', header = FALSE)
#test_matrix = as.matrix(test_data)
test_label=as.factor(unlist(test_data[,1])) #1st column value to factors
x_test =test_data[,-1]

########################## B-SPLINES ##########################
x = seq(0,1,length.out = dim(x_train)[2])
#set 8 knots
knots = seq(0,1,length.out = 8)
#find Bcoef in B-splines of training data
B = bs(x, knots = knots, degree = 3)[,1:10]
Bcoef = matrix(0,dim(x_train)[1],10)
for(i in 1:dim(x_train)[1])
{
  Bcoef[i,] = solve(t(B)%*%B)%*%t(B)%*%t(x_train[i,])
}
#fit the model with Bcoef in training data
fit = randomForest(train_label ~ .,
                   data=cbind.data.frame(as.data.frame(Bcoef),train_label))



#find Bcoef in B-splines of testing data
Bcoef_test = matrix(0,dim(x_test)[1],10)
for(i in 1:dim(x_test)[1])
{
  Bcoef_test[i,] = solve(t(B)%*%B)%*%t(B)%*%t(x_test[i,])
}
#based on Bcoef of testing data, prediction
pred = predict(fit,Bcoef_test)



#confusion matrix
bs_table=table(test_label,pred)
bs_table
#accurary
bs_accuracy = (bs_table[1] + bs_table[4])/sum(bs_table)
bs_accuracy
bs_false_positive=bs_table[2] /(bs_table[2]+bs_table[4])
bs_false_positive
bs_false_negative=bs_table[3] /(bs_table[3]+bs_table[1])
bs_false_negative

#plot all the samples in testing group 
#assume 1 is normal, -1 is abnormal
#plot B-spline Coefficients
#normal=blue;abnormal=red 
y1= x_test[pred == 1,]
matplot(x,t(y1),type="l",col = "blue",ylab = "ECG",main="Normal vs.Abnormal-B-spline Coefficients")

y2 = x_test[pred == -1,]
for(i in 1:length(pred[pred == -1]))
{
  lines(x,y2[i,],col = "red")
}
#plot oringinal label
#normal=green;abnormal=gray
y3= x_test[test_label==1,]
matplot(x,t(y3),type="l",col = "green",ylab = "ECG",main="Normal vs.Abnormal-original label")

y4 = x_test[test_label == -1,]
for(i in 1:length(test_label[test_label == -1]))
{
  lines(x,y4[i,],col = "gray")
}
#plot wrong predicted 

#normal but predicted as abnormal=magenta; abnormal but predicted as normal=black
y5= x_test[test_label==1&pred != 1,]
matplot(x,t(y5),type="l",col = "magenta",ylab = "ECG",main="Normal vs.Abnormal-wrong predicted(B-SPLINES)")


y6 = x_test[test_label == -1&pred != -1,]
for(i in 1:length(y6[,1]))
{
  lines(x,y6[i,],col = "black")
}


########################## FPCA ##########################
splinebasis = create.bspline.basis(c(0,1),10)
smooth = smooth.basis(x,t(x_train),splinebasis) #basis function
Xfun = smooth$fd
pca = pca.fd(Xfun, 10)
var.pca = cumsum(pca$varprop) #varprop--a vector giving the proportion of variance explained by each eigenfunction
nharm = sum(var.pca < 0.95) + 1
pc = pca.fd(Xfun, nharm)
plot(pc$scores[train_label==1,],xlab = "FPC-score normal", ylab = "FPC-score 2",col = "blue",ylim=c(-1,1))
points(pc$scores[train_label==-1,],col = "red")
FPCcoef = pc$scores
fit_FPCA = randomForest(train_label ~ .,
                   data=cbind.data.frame(as.data.frame(FPCcoef),train_label))


smooth_test = smooth.basis(x,t(x_test),splinebasis)
Xfun_test = smooth_test$fd
pca_test = pca.fd(Xfun_test, nharm)

FPCcoef_test = pca_test$scores
pred_FPCA = predict(fit_FPCA ,FPCcoef_test)
FPCA_table=table(test_label,pred_FPCA)
FPCA_table

FPCA_accuracy = (FPCA_table[1] + FPCA_table[4])/sum(FPCA_table)
FPCA_accuracy
FPCA_false_positive=FPCA_table[2] /(FPCA_table[2]+FPCA_table[4])
FPCA_false_positive
FPCA_false_negative=FPCA_table[3] /(FPCA_table[3]+FPCA_table[1])
FPCA_false_negative

#plot all the samples in testing group 
#assume 1 is normal, -1 is abnormal
#plot B-spline Coefficients
#normal=blue;abnormal=red 
y_1= x_test[pred_FPCA == 1,]
matplot(x,t(y_1),type="l",col = "blue",ylab = "ECG",main="Normal vs.Abnormal-FPCA")

y_2 = x_test[pred_FPCA == -1,]
for(i in 1:length(y_2))
{
  lines(x,y_2[i,],col = "red")
}


#plot wrong predicted 

#normal but predicted as abnormal=magenta; abnormal but predicted as normal=black
y_5= x_test[test_label==1&pred_FPCA != 1,]
matplot(x,t(y_5),type="l",col = "magenta",ylab = "ECG",main="Normal vs.Abnormal-wrong predicted (FPCA)")


y_6 = x_test[test_label == -1&pred_FPCA != -1,]
for(i in 1:length(y_6[,1]))
{
  lines(x,y_6[i,],col = "black")
}


