
#############################################
### kaggle competition
### bike share
### https://www.kaggle.com/c/bike-sharing-demand.
#############################################

# import data
train <- read.csv('../exercise/kaggle/bike/bike_train.csv', stringsAsFactors = F)
test <- read.csv('../exercise/kaggle/bike/bike_test.csv', stringsAsFactors = F)

# train
# add new features
train$datetime <- as.POSIXct(train$datetime)
train$year <- as.numeric(format(train$datetime, "%Y"))
train$month <- as.numeric(format(train$datetime, "%m"))
train$hour <- as.numeric(format(train$datetime, "%H"))
train$weekday <- factor(weekdays(train$datetime))
train$rush_hour <- train$workingday == 1 & 
  (train$hour %in% c(7, 8, 17, 18, 19))
train$adverse_rush_hour <- abs(train$hour - 14)

# test
# add new features
test$datetime <- as.POSIXct(test$datetime)
test$year <- as.numeric(format(test$datetime, "%Y"))
test$month <- as.numeric(format(test$datetime, "%m"))
test$hour <- as.numeric(format(test$datetime, "%H"))
test$weekday <- factor(weekdays(test$datetime))
test$rush_hour <- test$workingday == 1 & 
  (test$hour %in% c(7, 8, 17, 18, 19))
test$adverse_rush_hour <- abs(test$hour - 14)

##########################
# random forest
##########################
library(randomForest)

# predict count 
# 0.51158
# 0.48797 add weekday
rf <- randomForest(x = train[, c(1:9, 13:18)], y = train$count, 
                   importance = T)
rf
round(importance(rf), 2)
pred <- predict(rf, newdata = test)
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_rf_x2.csv", row.names = FALSE)

# predict casual and register separately
# 0.51606
rf_c <- randomForest(x = train[, c(1:9, 13:17)], y = train$casual, 
                     importance = T)
rf_c
round(importance(rf_c), 2)
pred_c <- predict(rf_c, newdata = test)
rf_r <- randomForest(x = train[, c(1:9, 13:17)], y = train$registered,
                     importance = T)
rf_r
round(importance(rf_r), 2)
pred_r <- predict(rf_r, newdata = test)
pred <- pred_c + pred_r
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_rf_x2_split.csv", row.names = FALSE)

###  log Y
# predict count 
# 0.43125
rf <- randomForest(x = train[, c(1:9, 13:17)], y = log(train$count), 
                   importance = T)
rf
round(importance(rf), 2)
pred <- predict(rf, newdata = test)
out <- data.frame('datetime'= test$datetime, 'count' = exp(pred))
write.csv(out, file = "../exercise/kaggle/bike/r_rf_x2_log.csv", row.names = FALSE)

# predict casual and register separately
# becuase zero exist in some value of casual or registered
# log(0) return -Inf, can not use random forest algorithm
# we should log(value + 1), then exp(pred) - 1
# 0.43005 x2
# 0.42802 x2 + x3
rf_c <- randomForest(x = train[, c(1:9, 13:19)], y = log(train$casual + 1), 
                     ntree = 1000, 
                     importance = T)
rf_c
round(importance(rf_c), 2)
pred_c <- predict(rf_c, newdata = test)
rf_r <- randomForest(x = train[, c(1:9, 13:19)], y = log(train$registered + 1),
                     ntree = 1000, 
                     importance = T)
rf_r
round(importance(rf_r), 2)
pred_r <- predict(rf_r, newdata = test)
pred <- exp(pred_c) + exp(pred_r) - 2
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_rf_x2+3_split_log+1.csv", row.names = FALSE)

##########################
# Extremely Randomized Trees
##########################
library(extraTrees)

# predict count 
# 0.50312
x_df <- train[, c(1:9, 13:17)]
x_df$datetime <- as.numeric(x_df$datetime)
x_df$rush_hour <- as.numeric(x_df$rush_hour)

x_test_df <- test
x_test_df$datetime <- as.numeric(x_test_df$datetime)
x_test_df$rush_hour <- as.numeric(x_test_df$rush_hour)

et <- extraTrees(x = as.matrix(x_df), y = train$count)
pred <- predict(et, newdata = as.matrix(x_test_df)) 
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_et_x2.csv", row.names = FALSE)

# log Y
######
### 0.43314
######
# predict count 
et <- extraTrees(x = as.matrix(x_df), y = log(train$count))
pred <- predict(et, newdata = as.matrix(x_test_df)) 
out <- data.frame('datetime'= test$datetime, 'count' = exp(pred))
write.csv(out, file = "../exercise/kaggle/bike/r_et_x2_log.csv", row.names = FALSE)

# predict casual and register separately
# memory.limit(102400)
# 0.43255
et_c <- extraTrees(x = as.matrix(x_df), y = log(train$casual + 1))
pred_c <- predict(et_c, newdata = as.matrix(x_test_df)) 
et_r <- extraTrees(x = as.matrix(x_df), y = log(train$registered + 1))
pred_r <- predict(et_r, newdata = as.matrix(x_test_df)) 
pred <- exp(pred_c) + exp(pred_r) - 2
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_et_x2_log_split.csv", row.names = FALSE)

# save(pred_c, x_test_df, file = "etree.RData")
# load("etree.RData")

##########################
# eXtreme Gradient Boosting
##########################
library(xgboost)

# split the train and test in train dataset
train$day <- as.numeric(format(train$datetime, "%d"))

# transform the data format
x_df <- train[, c(1:9, 13:18)]
x_df$datetime <- as.numeric(x_df$datetime)
x_df$rush_hour <- as.numeric(x_df$rush_hour)

train_test <- subset(x_df, day %in% c(16:19), select = -day)
train_train <- subset(x_df, day %in% c(1:15), select = -day)
# raw
dtrain <- xgb.DMatrix(data = as.matrix(train_train), 
                      label=train$count[train$day %in% c(1:15)])
dtest <- xgb.DMatrix(data = as.matrix(train_test),  
                     label=train$count[train$day %in% c(16:19)])
# log
dtrain <- xgb.DMatrix(data = as.matrix(train_train), 
                      label=log(train$count[train$day %in% c(1:15)]))
dtest <- xgb.DMatrix(data = as.matrix(train_test),  
                     label=log(train$count[train$day %in% c(16:19)])) 

x_test_df <- test 
x_test_df$datetime <- as.numeric(x_test_df$datetime) 
x_test_df$rush_hour <- as.numeric(x_test_df$rush_hour) 

# predict count
# 0.53450
params <- list(max_depth = 6, eta = .3, 
               objective = "reg:linear", eval_metric = "rmse") 
watchlist <- list(train = dtrain, test = dtest)
xgb <- xgb.train(params, 
                 dtrain, watchlist = watchlist,  
                 nrounds = 10)
# train-rmse:41.747948	test-rmse:53.609879
pred <- predict(xgb, as.matrix(x_test_df))
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_xgb_x2.csv", row.names = FALSE)

# predict count
#log Y
# 0.43474  the first time 
# 0.44147 train_train
# 0.44407 whole
params <- list(max_depth = 5, eta = .1, subsample = .9, 
               objective = "reg:linear", eval_metric = "rmse")
watchlist <- list(train = dtrain, test = dtest)
xgb <- xgb.train(params, 
                 dtrain, watchlist = watchlist,  
                 nrounds = 150)
# train-rmse:0.266304	test-rmse:0.355546 
# use the whole train data to train model
xgb <- xgb.train(params, 
                 xgb.DMatrix(as.matrix(x_df), label = log(as.matrix(train$count))), 
                 nrounds = 150)
pred <- predict(xgb, as.matrix(x_test_df))
out <- data.frame('datetime'= test$datetime, 'count' = exp(pred))
write.csv(out, file = "../exercise/kaggle/bike/r_xgb_x2_log.csv", row.names = FALSE)

# predict casual and register separately
dtrain <- xgb.DMatrix(data = as.matrix(train_train), 
                      label=log(train$casual[train$day %in% c(1:15)] + 1))
dtest <- xgb.DMatrix(data = as.matrix(train_test),  
                     label=log(train$casual[train$day %in% c(16:19)] + 1)) 
params <- list(max_depth = 5, eta = .1, subsample = .9, 
               objective = "reg:linear", eval_metric = "rmse")
watchlist <- list(train = dtrain, test = dtest)
bst0 <- xgb.train(params, 
                  dtrain, watchlist = watchlist,  
                  nrounds = 200)
# train-rmse:0.367365	test-rmse:0.539106

dtrain <- xgb.DMatrix(data = as.matrix(train_train), 
                      label=log(train$registered[train$day %in% c(1:15)] + 1))
dtest <- xgb.DMatrix(data = as.matrix(train_test),  
                     label=log(train$registered[train$day %in% c(16:19)] + 1)) 
params <- list(max_depth = 5, eta = .1, subsample = .9, 
               objective = "reg:linear", eval_metric = "rmse")
watchlist <- list(train = dtrain, test = dtest)
bst1 <- xgb.train(params, 
                  dtrain, watchlist = watchlist,  
                  nrounds = 200)
# train-rmse:0.222180	test-rmse:0.324818 
#log Y
# 0.43058
bst0 = xgb.train(params, xgb.DMatrix(as.matrix(x_df), label = log(as.matrix(train$casual) + 1)), 
                 nrounds = 200)
bst1 = xgb.train(params, xgb.DMatrix(as.matrix(x_df), label = log(as.matrix(train$registered) + 1)), 
                 nrounds = 200)
pred = exp(predict(bst0, as.matrix(x_test_df))) + 
  exp(predict(bst1, as.matrix(x_test_df))) - 2
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_xgb_x2_log_split.csv", row.names = FALSE)


#########################################
# https://www.analyticsvidhya.com/blog/2015/06/solution-kaggle-competition-bike-sharing-demand/

train <- read.csv('../exercise/kaggle/bike/bike_train.csv', stringsAsFactors = F)
test <- read.csv('../exercise/kaggle/bike/bike_test.csv', stringsAsFactors = F)

test$registered <- 0
test$casual <- 0
test$count <- 0

# could add new features once
# do not add train, then add test
data <- rbind(train, test)
# 
# str(data)
# 
# par(mfrow=c(4,2))
# par(mar = rep(2, 4))
# hist(data$season)
# hist(data$holiday)
# hist(data$workingday)
# hist(data$weather)
# hist(data$temp)
# hist(data$atemp)
# hist(data$humidity)
# hist(data$windspeed)
# table(data$weather)
# prop.table(table(data$weather))

# add new features
data$datetime <- as.POSIXct(data$datetime)
data$year <- as.numeric(format(data$datetime, "%Y"))
data$month <- as.numeric(format(data$datetime, "%m"))
data$hour <- as.numeric(format(data$datetime, "%H"))
data$weekday <- weekdays(data$datetime)

# back to train and test
# train_data <- data[as.integer(substr(data$datetime,9,10))<20,]
# test_data <- data[as.integer(substr(data$datetime,9,10))>19,]


####
## use decision tree to split variable "hour"
# library(rpart)
# library(rpart.plot)

# split hour for registered
# d <- rpart(registered ~ hour,data=train_data)
# rpart.plot(d)

data$hour_reg <- 0
data$hour_reg[data$hour < 7] = 1
data$hour_reg[data$hour == 7] = 2
data$hour_reg[data$hour == 8] = 3
data$hour_reg[data$hour > 8 & data$hour < 16] = 4
data$hour_reg[data$hour == 16 | data$hour == 17] = 5
data$hour_reg[data$hour == 18 | data$hour == 19] = 6
data$hour_reg[data$hour >= 20] = 7

# split hour for casual
# d_c <- rpart(casual ~ hour,data=train_data)
# rpart.plot(d_c)

data$hour_cas <- 0
data$hour_cas[data$hour <= 7] = 1
data$hour_cas[data$hour == 8 | data$hour == 9] = 2
data$hour_cas[data$hour >= 10 & data$hour < 20] = 3
data$hour_cas[data$hour >= 20] = 4

# split temperature for registered
# t_r <- rpart(registered ~ temp, data = train_data)
# rpart.plot(t_r)

data$temp_reg <- 0
data$temp_reg[data$temp < 13] = 1
data$temp_reg[data$temp >= 13 & data$temp < 23] = 2
data$temp_reg[data$temp >= 23 & data$temp < 30] = 3
data$temp_reg[data$temp >= 30] = 4

# split temperature for casual
# t_c <- rpart(casual ~ temp, data = train_data)
# rpart.plot(t_c)

data$temp_cas <- 0
data$temp_cas[data$temp < 15] = 1
data$temp_cas[data$temp >= 15 & data$temp < 23] = 2
data$temp_cas[data$temp >= 23 & data$temp < 30] = 3
data$temp_cas[data$temp >= 30] = 4

# split year into quarteres
data$year_part <- 0
data$year_part[data$year == 2011 & data$month <= 3] = 1
data$year_part[data$year == 2011 & data$month > 3 & data$month <= 6] = 2
data$year_part[data$year == 2011 & data$month > 6 & data$month <= 9] = 3
data$year_part[data$year == 2011 & data$month > 9] = 4
data$year_part[data$year == 2012 & data$month <= 3] = 5
data$year_part[data$year == 2012 & data$month > 3 & data$month <= 6] = 6
data$year_part[data$year == 2012 & data$month > 6 & data$month <= 9] = 7
data$year_part[data$year == 2012 & data$month > 9] = 8
# table(data$year_part)

# seperate workingday weekend holiday
data$day_type <- ""
data$day_type[data$holiday == 0 & data$workingday == 0] = "weekend"
data$day_type[data$holiday == 1] = "holiday"
data$day_type[data$holiday == 0 & data$workingday == 1] = "working day"

# created a variable for weekend
data$weekend <- 0
data$weekend[data$weekday == "星期六" | data$weekday == "星期日" ] = 1

# add old features
data$adverse_rush_hour <- abs(data$hour - 14)
data$rush_hour <- data$workingday == 1 & 
  (data$hour %in% c(7, 8, 17, 18, 19))

# transform to factor
data$season=as.factor(data$season)
data$weather=as.factor(data$weather)
data$holiday=as.factor(data$holiday)
data$workingday=as.factor(data$workingday)
data$year <- as.factor(data$year)
data$hour <- as.factor(data$hour)
data$month <- as.factor(data$month)
data$hour_reg <- as.factor(data$hour_reg)
data$hour_cas <- as.factor(data$hour_cas)
data$temp_reg <- as.factor(data$temp_reg)
data$temp_cas <- as.factor(data$temp_cas)

data$year_part <- as.factor(data$year_part)
data$weekend <- as.factor(data$weekend)
data$day_type <- as.factor(data$day_type)
data$weekday <- as.factor(data$weekday)
data$rush_hour <- as.factor(data$rush_hour)


# fill the missing value in windspeed by random forest
# table(data$windspeed == 0)
# k=data$windspeed==0
# wind_0=subset(data,k)
# wind_1=subset(data,!k)
# library(randomForest)
# set.seed(415)
# fit <- randomForest(windspeed ~ season+weather +humidity +month+temp+ year+atemp, 
#                     data=wind_1,importance=TRUE, ntree=250)
# pred=predict(fit,wind_0)
# wind_0$windspeed=pred
# data=rbind(wind_0,wind_1)
# 
# data <- data[order(as.numeric(rownames(data))), ]


# built the model 
train_data <- data[as.integer(substr(data$datetime,9,10))<20,]
test_data <- data[as.integer(substr(data$datetime,9,10))>19,]

# randomForest
# 0.41014 raw top5 features
# 0.43118 add "datetime", "temp", "month"
# 0.41103 add windspeed missing value and sort df
# 1.89367 to 0.41103
# top5 + adverse_rush_hour  0.40757
# top5 + adverse_rush_hour + rush_hour  0.40104
# add weekend 0.40173
library(randomForest)
set.seed(415)
# # for registered users
rf_r <- randomForest(x = train_data[, c("season", "holiday", "workingday",
                                        "weather", "atemp", "humidity",
                                        "windspeed", "year", "hour",
                                        "weekday", "hour_reg", "temp_reg",
                                        "year_part", "weekend", "day_type")],
                     y = log(train_data$registered + 1),
                     importance=TRUE, ntree=250)
rf_r
round(importance(rf_r), 2)
pred_r <- predict(rf_r, test_data[, c("season", "holiday", "workingday",
                                      "weather", "atemp", "humidity",
                                      "windspeed", "year", "hour",
                                      "weekday", "hour_reg", "temp_reg",
                                      "year_part", "weekend", "day_type")])

# for casual users
rf_c <- randomForest(x = train_data[, c("season", "holiday", "workingday",
                                        "weather", "atemp", "humidity",
                                        "windspeed", "year", "hour",
                                        "weekday", "hour_cas", "temp_cas",
                                        "year_part", "weekend", "day_type")],
                     y = log(train_data$casual + 1),
                     importance=TRUE, ntree=250)
rf_c
round(importance(rf_c), 2)
pred_c <- predict(rf_c, test_data[, c("season", "holiday", "workingday",
                                      "weather", "atemp", "humidity",
                                      "windspeed", "year", "hour",
                                      "weekday", "hour_cas", "temp_cas",
                                      "year_part", "weekend", "day_type")])
# 
# # combine predictors 
# pred <- exp(pred_c) + exp(pred_r) - 2
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_rf_x_top5_split.csv", row.names = FALSE)

##################################
library(extraTrees)
# predict casual and register separately
# memory.limit(102400)
# 0.43255
# 0.41305

x_r_df <- train_data[, c("season", "holiday", "workingday", 
                         "weather", "atemp", "humidity", 
                         "windspeed", "year", "hour", 
                         "weekday", "hour_reg", "temp_reg",
                         "year_part", "weekend", "day_type")]
for (i in 1:ncol(x_r_df)) {
  x_r_df[, i] <- as.numeric(x_r_df[, i])
}


x_c_df <- train_data[, c("season", "holiday", "workingday", 
                         "weather", "atemp", "humidity", 
                         "windspeed", "year", "hour", 
                         "weekday", "hour_cas", "temp_cas", 
                         "year_part", "weekend", "day_type")]
for (i in 1:ncol(x_c_df)) {
  x_c_df[, i] <- as.numeric(x_c_df[, i])
}


x_test_r_df <- test_data[, c("season", "holiday", "workingday", 
                             "weather", "atemp", "humidity", 
                             "windspeed", "year", "hour", 
                             "weekday", "hour_reg", "temp_reg", 
                             "year_part", "weekend", 
                             "day_type")]
for (i in 1:ncol(x_test_r_df)) {
  x_test_r_df[, i] <- as.numeric(x_test_r_df[, i])
}


x_test_c_df <- test_data[, c("season", "holiday", "workingday", 
                             "weather", "atemp", "humidity", 
                             "windspeed", "year", "hour", 
                             "weekday", "hour_cas", "temp_cas",
                             "year_part", "weekend", 
                             "day_type")]
for (i in 1:ncol(x_test_c_df)) {
  x_test_c_df[, i] <- as.numeric(x_test_c_df[, i])
}

et_c <- extraTrees(x = as.matrix(x_c_df), y = log(train_data$casual + 1))
pred_c <- predict(et_c, newdata = as.matrix(x_test_c_df)) 

# save(pred_c, train_data, test_data, test, file = "etree.RData")
# load("etree.RData")

et_r <- extraTrees(x = as.matrix(x_r_df), y = log(train_data$registered + 1))
pred_r <- predict(et_r, newdata = as.matrix(x_test_r_df)) 
pred <- exp(pred_c) + exp(pred_r) - 2
out <- data.frame('datetime'= test$datetime, 'count' = pred)
write.csv(out, file = "../exercise/kaggle/bike/r_et_x3.csv", row.names = FALSE)
















