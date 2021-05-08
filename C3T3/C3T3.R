#import dataset
#import libraries/install packages

install.packages("corrplot")
library(corrplot)
library(caret)


#create dummy variables
df <- dummyVars(" ~ .", data = data)

data <- data.frame(predict(df, newdata = data))

#view data structure
str(data) #everything is numeric, which is good for linear regression
#and for looking at correlation
summary(data) #there are 15 NAs under BestSellersRank

#remove NA values
data$BestSellersRank <- NULL #for now we are removing attribute with missing data

#rename columns
names(data)<-c("Accessories","Display","Warranty", "Game", "Laptop", "Notebook",
               "PC", "Printer", "PrintSupplies", "SmartPhone", "Software", "Tablet",
               "ProductNum", "Price", "fivestar", "fourstar", "threestar", "twostar",
               "onestar", "PosReview", "NegReview", "Recommend", "ShipWt",
               "Depth", "Width", "Height", "Profit", "Volume") 
#for this write a name for every column or it will list it as NA


#view correlation between features
corrData <- cor(data)
corrData

#plot corr matrix
corrplot(corrData) #saved plot as png so its easier to read

#create linear model
#set seed first
set.seed(123)

#create 75%/25% train/test split of dataset
inTraining <- createDataPartition(data$Volume, p = .75, list = FALSE)
training <- data[inTraining,]
testing <- data[-inTraining,]

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train Linear Regression model
LMFit1 <- train(Volume~. -ProductNum, data = training, method = "lm", 
                trControl = fitControl)

LMFit1
summary(LMFit1)

#model has R squared of 1
#data is non-parametric -- will try different model type

##########################################################################

#SVM model

set.seed(123)
svmFit1 <- train(Volume~. -ProductNum, data = training, method = "svmLinear",
                 trControl = fitControl)
svmFit1
summary(svmFit1)

svmpredict1 <- predict(svmFit1, testing)
svmpredict1

plot(svmpredict1)

plot(testing$Volume)
points(svmpredict1, col = "blue", pch = 4)

#result: RMSE = 198.2904, R2 =  0.917478, MAE =  140.8567, C held at 1


#SVM 2 (try using tuning grid with Cost)
#tried to center/scale data -- 
#warning message: these variables have zero variances
grid <- expand.grid(C = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 
                          1.25, 1.5, 1.75, 2,5))
set.seed(123)
svmFit2 <- train(Volume~. -ProductNum, data = training, method = "svmLinear",
                 trControl = fitControl, tuneGrid = grid)
svmFit2

#error shows 38 warnings...
warnings()

svmpredict2 <- predict(svmFit2, testing)
svmpredict2

postResample(svmpredict2, testing$Volume)

varImp(svmFit2)

plot(testing$Volume)
points(svmpredict2, col = "blue", pch = 4)

#svm best result
# C = 0.10, RMSE  = 177.7844, R2  = 0.9546032, MAE =  124.9061

#postResample
#    RMSE        Rsquared    MAE 
# 547.1429196   0.8060408 286.8606898 

# with 7 as seed
# C = 0.50, RMSE = 123.3788, R2 = 0.9672854, MAE  = 88.23432


#try eliminating features
data3 <- data[-16] #eliminated 4stars since its so highly correlated to 5star
#tried also eliminating 3star and posreview, but didn't work as well

#create 75%/25% train/test split of dataset
set.seed(123)
inTraining3 <- createDataPartition(data3$Volume, p = .75, list = FALSE)
training3 <- data3[inTraining3,]
testing3 <- data3[-inTraining3,]

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 
                          1.25, 1.5, 1.75, 2, 5))
set.seed(123)
svmFit3 <- train(Volume~. -ProductNum, data = training3, method = "svmLinear",
                 trControl = fitControl, tuneGrid = grid)
svmFit3

svmpredict3 <- predict(svmFit3, testing3)
svmpredict3


#Results: C = 0.25, RMSE =  143.2764, R2 =  0.9618955, MAE =  119.4248
#better results but getting negative values...

postResample(svmpredict3, testing3$Volume)

#RMSE = 514.8320434, R2 = 0.8141355, MAE = 278.0926113 

varImp(svmFit3)

#########################################################################

#RF model

set.seed(123)
rfFit1 <- train(Volume~. -ProductNum, data = training, method = "rf",
                trControl = fitControl)
rfFit1

rfpredict <- predict(rfFit1, testing)
rfpredict

plot(testing$Volume)
points(rfpredict, col = "blue", pch = 4)

postResample(rfpredict, testing$Volume)

# result: mtry = 26,  RMSE = 703.2548, R2 =  0.9600554, MAE = 305.6153

set.seed(123)
rfgrid <- expand.grid(mtry = c(1,2,3,4,5))
rfFit2 <- train(Volume~. -ProductNum, data = training, method = "rf",
                trControl = fitControl, tuneGrid = rfgrid)
rfFit2

varImp(rfFit2)

#result: mtry = 5, RMSE = 740.4537, R2 =  0.9217604, MAE =  354.0184

set.seed(123)
rfFit3 <- train(Volume~. -ProductNum, data = training3, method = "rf",
                trControl = fitControl)
rfFit3
#tried with an eliminated feature (4star). Results for RF overall not great.

varImp(rfFit3)

#######################################################################

#Gradient Boosting

set.seed(123)
gbFit1 <- train(Volume~. -ProductNum, data = training, method = "BstLm",
                trControl = fitControl)
gbFit1

gbpredict <- predict(gbFit1, testing)
gbpredict

postResample(gbpredict, testing$Volume)

varImp(gbFit1)

plot(testing$Volume)
points(gbpredict, col = "blue", pch = 4)

#results: mstop = 150, RMSE  = 173.1271, R2 =  0.9434552, MAE =  133.1898
#still neg values in predictions

set.seed(123) #trying with eliminated 4star
gbFit2 <- train(Volume~. -ProductNum, data = training3, method = "BstLm",
                trControl = fitControl)
gbFit2

gbpredict2 <- predict(gbFit2, testing3)
gbpredict2

#results: mstop = 150,  RMSE =  173.1271, R2 =  0.9434552, MAE =  133.1898

#######################################################################

#look at data distribution of training and testing sets

hist(training$Volume)
hist(testing$Volume)


hist(training$fivestar)
hist(testing$fivestar)


#######################################################################

#import new products dataset

df2 <- dummyVars(" ~ .", data = newdata)

newproducts <- data.frame(predict(df2, newdata = newdata))

newproducts$BestSellersRank <- NULL

names(newproducts)<-c("Accessories","Display","Warranty", "Game", "Laptop", "Notebook",
               "PC", "Printer", "PrintSupplies", "SmartPhone", "Software", "Tablet",
               "ProductNum", "Price", "fivestar", "fourstar", "threestar", "twostar",
               "onestar", "PosReview", "NegReview", "Recommend", "ShipWt",
               "Depth", "Width", "Height", "Profit", "Volume") 

newpredictions <- predict(gbFit1, newproducts)
newpredictions

#add new predictions to the dataframe
newproducts$newpreds <- predict(gbFit1, newproducts)

#save the dataset with added predictions as a .csv file
write.table(newproducts, file = "newproductspreds.csv",
            sep = ",", row.names = F)
