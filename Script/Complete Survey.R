pacman::p_load("readr", "caret", "mlbench", "rpart", "rpart.plot", "party",
               "randomForest", "C50", "inum", "RColorBrewer", "bindrcpp")
getwd()
dt <- read.csv("CompleteResponses.csv")

attributes(dt)
summary(dt)
str(dt)

#### Pre-processing the data #####
dt$brand<-as.factor(dt$brand)
dt$elevel<-as.factor(dt$elevel)
dt$car<-as.factor(dt$car)
dt$zipcode<-as.factor(dt$zipcode)
boxplot(dt$salary)
boxplot(dt$credit)
names(dt)<-c("Salary","Age","Education_Level", "Car", "Zip_Code","Credit",
             "Brand")

dt$Education_Level<- sub(x = dt$Education_Level,pattern = 0, 
    replacement =  "Less than High School Degree")
dt$Education_Level<- sub(x = dt$Education_Level, pattern = 1, 
    replacement =  "High School Degree")
dt$Education_Level<- sub(x = dt$Education_Level,pattern = 2, 
    replacement =  "Some College")
dt$Education_Level<- sub(x = dt$Education_Level,pattern = 4,
    replacement="Master's, Doctoral or Professional Degree")
dt$Education_Level<- sub(x = dt$Education_Level,pattern = 3, 
    replacement =  "4-Year College Degree")

dt$Car<- sub(x = dt$Car, pattern = 20, replacement = "None of the above")
dt$Car<- sub(x = dt$Car, pattern = 19, replacement = "Toyota")
dt$Car<- sub(x = dt$Car, pattern = 18, replacement = "Subaru")
dt$Car<- sub(x = dt$Car, pattern = 17, replacement = "Ram")
dt$Car<- sub(x = dt$Car, pattern = 16, replacement = "Nissan")
dt$Car<- sub(x = dt$Car, pattern = 15, replacement = "Mitsubishi")
dt$Car<- sub(x = dt$Car, pattern = 14, replacement = "Mercedes Benz")
dt$Car<- sub(x = dt$Car, pattern = 13, replacement = "Mazda")
dt$Car<- sub(x = dt$Car, pattern = 12, replacement = "Lincoln")
dt$Car<- sub(x = dt$Car, pattern = 11, replacement = "Kia")
dt$Car<- sub(x = dt$Car, pattern = 10, replacement = "Jeep")
dt$Car<- sub(x = dt$Car, pattern = 9, replacement = "Hyundai")
dt$Car<- sub(x = dt$Car, pattern = 8, replacement = "Honda")
dt$Car<- sub(x = dt$Car, pattern = 7, replacement = "Ford")
dt$Car<- sub(x = dt$Car, pattern = 6, replacement = "Dodge")
dt$Car<- sub(x = dt$Car, pattern = 5, replacement = "Chrysler")
dt$Car<- sub(x = dt$Car, pattern = 4, replacement = "Chevrolet")
dt$Car<- sub(x = dt$Car, pattern = 3, replacement = "Cadillac")
dt$Car<- sub(x = dt$Car, pattern = 2, replacement = "Buick")
dt$Car<- sub(x = dt$Car, pattern = 1, replacement = "BMW")

dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=0, replacement="New England")
dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=1, replacement="Mid-Atlantic")
dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=2, replacement="East North Central")
dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=3, replacement="West North Central")
dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=4, replacement="South Atlantic")
dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=5, replacement="East South Central")
dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=6, replacement="West South Central")
dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=7, replacement="Mountain")
dt$Zip_Code<- sub(x=dt$Zip_Code, pattern=8, replacement="Pacific")

dt$Brand<- sub(x=dt$Brand, pattern=0, replacement="Acer")
dt$Brand<- sub(x=dt$Brand, pattern=1, replacement="Sony")

#### Plotting ####
# As we can observe at below scatterplot, there are a strong relation on which brand will preffer our customer, depending at their age and their salary.
ggplot(dt,aes(x=Salary,y=Age,color=Brand)) + geom_point() + geom_smooth()

ggplot(data=dt, aes (x= Car, fill=Brand)) + geom_bar() +
  ggtitle(label = "Barplot Car") + theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(data=dt, aes(x=Age, fill=Brand))+geom_bar() +
  ggtitle(label = "Barplot Age") + theme_bw()

ggplot(data= dt, aes(x=Education_Level, fill=Brand)) + geom_bar() + 
  aes(stringr::str_wrap(Education_Level,15)) + xlab(NULL) +
  ggtitle(label = "Barplot Education Level") + theme_bw()

ggplot(data=dt, aes (x= Zip_Code, fill=Brand))+ geom_bar() + 
  aes(stringr::str_wrap(Zip_Code,15)) + xlab(NULL) + 
  ggtitle(label = "Barplot Zip Code") + theme_bw()

#### Chi-squared Tests ####

chisq.test(x=dt$Brand , y=dt$Education_Level)
chisq.test(x=dt$Brand , y=dt$Car)
chisq.test(x=dt$Brand , y=dt$Zip_Code)

chisq.test(x=dt$Education_Level , y=dt$Car)
chisq.test(x=dt$Education_Level , y=dt$Zip_Code)

chisq.test(x=dt$Car , y=dt$Zip_Code)

#### Decision Trees ####
TreeAnalysis <- rpart(Brand ~ ., data=dt)
rpart.plot(TreeAnalysis,extra=1)

tree <- ctree(dt$Brand~., data=dt, 
              controls = ctree_control(mincriterion = 0.95, minsplit = 3000))

dt1 <- ctree(dt$Brand~., data=dt , 
             controls = ctree_control(maxdepth = 3))

plot(dt1)

#### Creating a Model ####
set.seed(123)
inTrain <- createDataPartition(y = dt$Brand,
                               p = .75,
                               list = FALSE)

str(inTrain)

training <- dt[ inTrain,]
testing <- dt[-inTrain,]
nrow(training)
nrow(testing)

# k-NN model
set.seed(123)
knnFit <- train(Brand ~ Salary+Age,
                data = training,
                method = "knn",
                tuneLength = 15,
                metric = "Accuracy",
                preProc = c("center", "scale"))
knnFit
plot(knnFit)

knnClasses <- predict(knnFit, newdata = testing)
str(knnClasses)
plot(knnClasses)

# Random Forest model
set.seed(123)
ctrl <- trainControl(method = "cv",
                     number = 10,
                     search = "grid")

rfFit <- train(Brand ~ Salary+Age,
               data = training,
               method = "rf",
               metric = "Accuracy",
               preProc = c("center", "scale"),
               tuneLength = 15,
               trControl = ctrl)
rfFit

rfClasses <- predict(rfFit, newdata = testing)
str(rfClasses)
plot(rfClasses)

# C5.0
set.seed(849)
tree_mod <- train(Brand ~ Salary+Age,
                  data = training,
                  method = "C5.0", 
                  trControl = ctrl,
                  preProc = c("center", "scale"),
                  tuneLength = 15)

tree_mod
plot(tree_mod)

tree_Classes <- predict(tree_mod, newdata = testing)
str(tree_Classes)
plot(tree_Classes)

fitted(tree_Classes)


#### Model using the whole data set ####
tree_mod_2 <- train(Brand ~ Salary+Age,
               data = dt,
               method = "C5.0", 
               trControl = ctrl,
               preProc = c("center", "scale"),
               tuneLength = 15)

tree_mod_2
plot(tree_mod_2)

#### Incomplete Survey load and pre-processing ####
dt_2 <- read.csv("SurveyIncomplete.csv")

dt_2$brand<-as.factor(dt_2$brand)
dt_2$elevel<-as.factor(dt_2$elevel)
dt_2$car<-as.factor(dt_2$car)
dt_2$zipcode<-as.factor(dt_2$zipcode)
names(dt_2)<-c("Salary","Age","Education_Level", "Car", "Zip_Code","Credit",
             "Brand")

dt_2$Education_Level<- sub(x = dt_2$Education_Level,pattern = 0, 
                         replacement =  "Less than High School Degree")
dt_2$Education_Level<- sub(x = dt_2$Education_Level, pattern = 1, 
                         replacement =  "High School Degree")
dt_2$Education_Level<- sub(x = dt_2$Education_Level,pattern = 2, 
                         replacement =  "Some College")
dt_2$Education_Level<- sub(x = dt_2$Education_Level,pattern = 4,
                         replacement="Master's, Doctoral or Professional Degree")
dt_2$Education_Level<- sub(x = dt_2$Education_Level,pattern = 3, 
                         replacement =  "4-Year College Degree")

dt_2$Car<- sub(x = dt_2$Car, pattern = 20, replacement = "None of the above")
dt_2$Car<- sub(x = dt_2$Car, pattern = 19, replacement = "Toyota")
dt_2$Car<- sub(x = dt_2$Car, pattern = 18, replacement = "Subaru")
dt_2$Car<- sub(x = dt_2$Car, pattern = 17, replacement = "Ram")
dt_2$Car<- sub(x = dt_2$Car, pattern = 16, replacement = "Nissan")
dt_2$Car<- sub(x = dt_2$Car, pattern = 15, replacement = "Mitsubishi")
dt_2$Car<- sub(x = dt_2$Car, pattern = 14, replacement = "Mercedes Benz")
dt_2$Car<- sub(x = dt_2$Car, pattern = 13, replacement = "Mazda")
dt_2$Car<- sub(x = dt_2$Car, pattern = 12, replacement = "Lincoln")
dt_2$Car<- sub(x = dt_2$Car, pattern = 11, replacement = "Kia")
dt_2$Car<- sub(x = dt_2$Car, pattern = 10, replacement = "Jeep")
dt_2$Car<- sub(x = dt_2$Car, pattern = 9, replacement = "Hyundai")
dt_2$Car<- sub(x = dt_2$Car, pattern = 8, replacement = "Honda")
dt_2$Car<- sub(x = dt_2$Car, pattern = 7, replacement = "Ford")
dt_2$Car<- sub(x = dt_2$Car, pattern = 6, replacement = "Dodge")
dt_2$Car<- sub(x = dt_2$Car, pattern = 5, replacement = "Chrysler")
dt_2$Car<- sub(x = dt_2$Car, pattern = 4, replacement = "Chevrolet")
dt_2$Car<- sub(x = dt_2$Car, pattern = 3, replacement = "Cadillac")
dt_2$Car<- sub(x = dt_2$Car, pattern = 2, replacement = "Buick")
dt_2$Car<- sub(x = dt_2$Car, pattern = 1, replacement = "BMW")

dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=0, replacement="New England")
dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=1, replacement="Mid-Atlantic")
dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=2, replacement="East North Central")
dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=3, replacement="West North Central")
dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=4, replacement="South Atlantic")
dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=5, replacement="East South Central")
dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=6, replacement="West South Central")
dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=7, replacement="Mountain")
dt_2$Zip_Code<- sub(x=dt_2$Zip_Code, pattern=8, replacement="Pacific")

#### Applying the model to Incomplete Survey data set ####

tree_Classes_2 <- predict(tree_mod_2, newdata = dt_2)
str(tree_Classes_2)
plot(tree_Classes_2)

dt_2$Brand <- predict(tree_mod_2, newdata = dt_2)

#### Combining both data sets ####
aggr_dt <- rbind(dt,dt_2)

ggplot(data=aggr_dt, aes(x=Brand)) + geom_bar(aes(x=Brand, fill=Brand)) + 
  ggtitle(label = "Total Surveys")


#### Plotting Total Survey ####
dt$Survey <- c("Complete")
dt_2$Survey <- c("Incomplete")
aggr_dt <- rbind(dt,dt_2)
  
ggplot(aggr_dt, aes(x=Education_Level, colour=Survey)) + geom_density() + 
  aes(stringr::str_wrap(Education_Level,15)) + xlab(NULL) +
  ggtitle(label = "Densityplot Education Level")

ggplot(aggr_dt, aes(x=Car, colour=Survey)) + geom_density() +
  ggtitle(label = "Densityplot Car") + theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(aggr_dt, aes(x=Zip_Code, colour=Survey)) + geom_density() + 
  aes(stringr::str_wrap(Zip_Code,15)) + xlab(NULL) +
  ggtitle(label = "Densityplot Zip Code")

ggplot(aggr_dt, aes(x=Brand, colour=Survey)) + geom_density() +
  ggtitle(label = "Densityplot Brand") + theme_bw()


#### Original DataSet ####
Original_dt <- read_csv("Blackwell_Hist_Sample.csv")

names(Original_dt)<-c("in_store","age","items", "amount", "region")

Original_dt$region<- sub(x = Original_dt$region,pattern = 1, 
                         replacement =  "East")
Original_dt$region<- sub(x = Original_dt$region,pattern = 2, 
                         replacement =  "West")
Original_dt$region<- sub(x = Original_dt$region,pattern = 3, 
                         replacement =  "South")
Original_dt$region<- sub(x = Original_dt$region,pattern = 4, 
                         replacement =  "Central")

Original_dt$in_store<- sub(x = Original_dt$in_store,pattern = 0, 
                           replacement =  "Online")
Original_dt$in_store<- sub(x = Original_dt$in_store,pattern = 1, 
                         replacement =  "In-store")
# Ploting
ggplot(data=Original_dt, aes(x=age, fill= region))+
  geom_bar() + ggtitle(label = "Barplot Age from Original Sample") + theme_bw()

ggplot(data=Original_dt, aes (x= region))+ geom_bar() + 
  ggtitle(label = "Barplot Zip Code from Original Sample") + theme_bw()
