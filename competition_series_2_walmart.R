
#############################################
### kaggle competition
### walmart store sales forecasting
### https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
### https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/leaderboard
### https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8125
#############################################
# edited on 20170706

######################
### simple model
### weighted mean absolute error 
### (wmae): 3455.81209
### https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8033
#######################
# import data
train <- read.csv('../exercise/kaggle/walmart/train.csv', stringsAsFactors = F)
test <- read.csv('../exercise/kaggle/walmart/test.csv', stringsAsFactors = F)

# transform date and create some variables
train$Date <- as.Date(train$Date)
train$week_of_year <- as.numeric(format(train$Date, format = "%U"))
train$year <- as.numeric(format(train$Date, format = "%Y"))

test$Date <- as.Date(test$Date)
test$week_of_year <- as.numeric(format(test$Date, format = "%U"))
test$year <- as.numeric(format(test$Date, format = "%Y"))
test$m_year <- test$year - 1

# match by store, dept, year and week_of_year
out <- merge(train[, c("Store", "Dept", "Weekly_Sales", "year", "week_of_year")], 
             test[, c("Store", "Dept", "m_year", "week_of_year", "Date", "IsHoliday")], 
             by.x = c("Store", "Dept", "year", "week_of_year"), 
             by.y = c("Store", "Dept", "m_year", "week_of_year"), 
             all.y = TRUE)

# fill the missing value in 0
out$Weekly_Sales[is.na(out$Weekly_Sales)] <- 0

# create Id col
out$Id <- paste(out$Store, out$Dept, out$Date, sep = "_")
# export data 
write.csv(out[, c("Id", "Weekly_Sales")], 
          '../exercise/kaggle/walmart/simple_model_20170704.csv', 
          row.names = F)
### weighted mean absolute error 
### (wmae): 3455.81209

# the function seasonal.naive
# the same with above codes merge
# which is used in function 'grouped.forecast'
# 2012-11-02 value was predicted by 2011-11-04, the 44th weeks in the year
# which is before one year the same week
seasonal.naive <- function(train, test){
  # Computes seasonal naive forecasts
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  h <- nrow(test)
  tr <- train[nrow(train) - (52:1) + 1,]
  tr[is.na(tr)] <- 0
  test[,2:ncol(test)]  <- tr[1:h,2:ncol(test)]
  test
}



######################
### SVD + stlf/ets
### weighted mean absolute error 
### (wmae): 
### https://github.com/davidthaler/Walmart_competition_code/blob/master/grouped.forecast.R
######################

library(forecast)
library(plyr)
library(reshape)

paths <- list(data = '../exercise/kaggle/walmart/', 
              submit = '../exercise/kaggle/walmart/submissions/')

### import data
train <- read.csv(paste0(paths$data, 'train.csv'), 
                  colClasses = c('factor', 'factor', 'Date', 'numeric', 'logical'))
test <- read.csv(paste0(paths$data, 'test.csv'), 
                 colClasses = c('factor', 'factor', 'Date', 'logical'))

# the function forecast
grouped.forecast <- function(train, test, fname, ...) {
  
  f <- get(fname)
  
  ### test
  test_dates <- unique(test$Date)
  num_test_dates <- length(test_dates)
  
  all_stores <- unique(test$Store)
  num_stores <- length(all_stores)
  
  test_depts <- unique(test$Dept)
  #reverse the depts so the grungiest data comes first
  test_depts <- test_depts[length(test_depts):1]
  
  forecast_frame <- data.frame(Date=rep(test_dates, num_stores),
                               Store=rep(all_stores, each=num_test_dates))
  pred <- test
  pred$Weekly_Sales <- 0
  
  
  ### train
  train_dates <- unique(train$Date)
  num_train_dates <- length(train_dates)
  
  train_frame <- data.frame(Date=rep(train_dates, num_stores),
                            Store=rep(all_stores, each=num_train_dates))
  
  ## prediction
  for(d in test_depts){
    # for every dept
    print(paste('dept:', d))
    
    tr_d <- train_frame
    # This joins in Weekly_Sales but generates NA's. Resolve NA's 
    # in the model because they are resolved differently in different models.
    tr_d <- join(x = tr_d, y = train[train$Dept==d, c('Store','Date','Weekly_Sales')])
    tr_d <- cast(tr_d, Date ~ Store)    
    
    fc_d <- forecast_frame
    fc_d$Weekly_Sales <- 0
    fc_d <- cast(fc_d, Date ~ Store)
    
    result <- f(tr_d, fc_d, ...)
    # This has all Stores/Dates for this dept, but may have some that
    # don't go into the submission.
    result <- melt(result)
    pred.d.idx <- pred$Dept==d
    #These are the Store-Date pairs in the submission for this dept
    pred.d <- pred[pred.d.idx, c('Store', 'Date')]
    pred.d <- join(pred.d, result)
    pred$Weekly_Sales[pred.d.idx] <- pred.d$value
  }
  
  pred
}

# the function stlf.svd
# which is used in function 'grouped.forecast'
stlf.svd <- function(train, test, model.type, n.comp){
  # Replaces the training data with a rank-reduced approximation of itself,
  # then forecasts each store using stlf() from the forecast package.
  # That function performs an STL decomposition on each series, seasonally
  # adjusts the data, non-seasonally forecasts the seasonally adjusted data,
  # and then adds in the naively extended seasonal component to get the
  # final forecast.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  # model.type - one of 'ets' or 'arima', specifies which type of model to
  #        use for the non-seasonal forecast
  # n.comp - the number of components to keep in the singular value
  #         decomposition that is performed for preprocessing
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  horizon <- nrow(test)
  train <- preprocess.svd(train, n.comp) 
  for(j in 2:ncol(train)){
    # transform to time series, frequency is 52, one year has 52 weeks
    # for every store
    s <- ts(train[, j], frequency=52)
    if(model.type == 'ets'){
      fc <- stlf(s, 
                 h=horizon, # Number of periods for forecasting
                 # Either the character string ¡°periodic¡± or 
                 # the span (in lags) of the loess window for seasonal extraction
                 s.window=3, 
                 method='ets',
                 ic='bic', 
                 opt.crit='mae')
    }else if(model.type == 'arima'){
      fc <- stlf(s, 
                 h=horizon, 
                 s.window=3, 
                 method='arima',
                 ic='bic')
    }else{
      stop('Model type must be one of ets or arima.')
    }
    pred <- as.numeric(fc$mean)
    test[, j] <- pred
  }
  test
}

# the function singular value decomposition
# which is used in function 'stlf.svd'
preprocess.svd <- function(train, n.comp){
  # Replaces the training data with a rank-reduced approximation of itself.
  # This is for noise reduction. The intuition is that characteristics
  # that are common across stores (within the same department) are probably
  # signal, while those that are unique to one store may be noise.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # n.comp - the number of components to keep in the singular value
  #         decomposition
  #
  # returns:
  #  the rank-reduced approximation of the training data
  train[is.na(train)] <- 0
  z <- svd(train[, 2:ncol(train)], nu=n.comp, nv=n.comp)
  s <- diag(z$d[1:n.comp])
  train[, 2:ncol(train)] <- z$u %*% s %*% t(z$v)
  train
}

### SVD + stlf/ets
pred <- grouped.forecast(train, test, 'stlf.svd', model.type='ets', n.comp=12)

# creat submission file 
out <- data.frame('Id' = paste(pred$Store, pred$Dept, pred$Date, sep = "_"), 
                  'Weekly_Sales' = pred$Weekly_Sales)
# export data 
write.csv(out, 
          '../exercise/kaggle/walmart/submissions/svd_ets_20170706.csv', 
          row.names = F)
### weighted mean absolute error 
### (wmae): 2626,21649
########################

# the function postprocess
# similar to function grouped.forecast
# could add fname 'shift'
postprocess <- function(train, test, ...){
  # Iterates over the departments and calls shift() on each.
  #
  # args:
  #  train - the training set as returned from raw.train() in util 
  #  test - a reloaded submission or a data frame similar to test,
  #         from raw.test() in util, but with predictions in the 
  #         Weekly_Sales field
  # ... - additional arguments passed to shift()
  #
  # returns:
  #  the data frame input as test, after calling shift on it department-wise
  if('Id' %in% names(test)){
    #This is a saved submission
    sales <- test$Weekly_Sales
    test <- raw.test()
    test$Weekly_Sales <- sales
  }
  
  # test
  test.dates <- unique(test$Date)
  num.test.dates <- length(test.dates)
  
  all.stores <- unique(test$Store)
  num.stores <- length(all.stores)
  
  test.depts <- unique(test$Dept)
  
  forecast.frame <- data.frame(Date=rep(test.dates, num.stores),
                               Store=rep(all.stores, each=num.test.dates))
  pred <- test
  pred$Weekly_Sales <- 0
  
  # train
  train.dates <- unique(train$Date)
  num.train.dates <- length(train.dates)
  
  train.frame <- data.frame(Date=rep(train.dates, num.stores),
                            Store=rep(all.stores, each=num.train.dates))
  
  for(d in test.depts){
    # for every dept
    print(paste('dept:', d))
    
    tr.d <- join(train.frame,
                 train[train$Dept==d, c('Store','Date','Weekly_Sales')])
    tr.d <- cast(tr.d, Date ~ Store) 
    
    fc.d <- join(forecast.frame,
                 test[test$Dept==d, c('Store', 'Date', 'Weekly_Sales')])
    fc.d <- cast(fc.d, Date ~ Store)
    
    result <- shift(tr.d, fc.d, ...)
    
    result <- melt(result)
    pred.d.idx <- pred$Dept==d
    pred.d <- pred[pred.d.idx, c('Store', 'Date')]
    pred.d <- join(pred.d, result)
    pred$Weekly_Sales[pred.d.idx] <- pred.d$value
  }
  pred
}

# the function shift
# which is used in function 'postprocess'
shift <- function(train, test, threshold=1.1, shift=2){
  # This function executes a shift of the sales forecasts in the Christmas
  # period to reflect that the models are weekly, and that the day of the week
  # that Christmas occurs on shifts later into the week containing the holiday.
  #
  # NB: Train is actually not used here. Previously, there were other post-
  #     adjustments which did use it, and it is taken in here to preserve a 
  #     calling signature.
  #
  # args:
  # train - this is an n_weeks x n_stores matrix of values of Weekly_Sales
  #         for the training set within department, across all the stores
  # test - this is a (forecast horizon) x n_stores matrix of Weekly_Sales
  #        for the training set within department, across all the stores
  # threshold - the shift is executed if the mean of Weekly_Sales for weeks
  #          49-51 is greater than that for weeks 48 and 52 by at least
  #          a ratio of threshold
  # shift - The number of days to shift sales around Christmas.
  #         Should be 2 if the model is based on the last year only,
  #         or 2.5 if it uses both years
  #
  # returns:
  #  the test data 
  s <- ts(rep(0,39), frequency=52, start=c(2012,44))
  idx <- cycle(s) %in% 48:52
  holiday <- test[idx, 2:46]
  baseline <- mean(rowMeans(holiday[c(1, 5), ], na.rm=TRUE))
  surge <- mean(rowMeans(holiday[2:4, ], na.rm=TRUE))
  holiday[is.na(holiday)] <- 0
  
  if(is.finite(surge/baseline) & surge/baseline > threshold){
    shifted.sales <- ((7-shift)/7) * holiday
    shifted.sales[2:5, ] <- shifted.sales[2:5, ] + (shift/7) * holiday[1:4, ]
    shifted.sales[1, ] <- holiday[1, ]
    test[idx, 2:46] <- shifted.sales
  }
  test
}

### 
pred <- postprocess(train, pred, shift=2.5)

# creat submission file 
out <- data.frame('Id' = paste(pred$Store, pred$Dept, pred$Date, sep = "_"), 
                  'Weekly_Sales' = pred$Weekly_Sales)
# export data 
write.csv(out, 
          '../exercise/kaggle/walmart/submissions/svd_ets_shift_20170706.csv', 
          row.names = F)
### weighted mean absolute error 
### (wmae): 2321.43150


# This is model 1 from the post. It gets 2348 on the final board.
pred <- grouped.forecast(train, test, 'stlf.svd', model.type='ets', n.comp=12)
pred <- postprocess(train, pred, shift=2.5)
s.num <- write.submission(pred)
sub.nums <- c(sub.nums, s.num)

# This is model 2 from the post.
pred <- grouped.forecast(train, test, 'stlf.svd', model.type='arima', n.comp=12)
pred <- postprocess(train, pred, shift=2.5)
s.num <- write.submission(pred)
sub.nums <- c(sub.nums, s.num)



######################
### other two simple models
### product and 
### linear regression of time series
### https://github.com/davidthaler/Walmart_competition_code/blob/master/grouped.forecast.R
######################

# the simple model 2
product <- function(train, test){
  # Computes forecasts with the product model. This model predicts the mean
  # value by store times the mean value by week divided by the mean value
  # over the department.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  h <- nrow(test)
  # the value of last year
  # the outcome of function 'seasonal.naive'
  tr <- train[nrow(train) - (52:1) + 1,]
  tr[is.na(tr)] <- 0
  # compute one dept one store's mean from 2011-11-04 to 2012-10-26
  levels <- colMeans(tr[,2:ncol(tr)])
  # compute one dept all store's mean on one day such as 2011-11-04
  profile <- rowMeans(tr[,2:ncol(tr)])
  
  overall <- mean(levels)
  # cell 11: mean(store1_allDay)*mean(2011-11-04_allStore)/mean(total_table)
  # cell 12: mean(store1_allDay)*mean(2011-11-11_allStore)/mean(total_table)
  pred <- matrix(profile, ncol=1) %*% matrix(levels, nrow=1)
  pred <- pred / overall
  test[,2:ncol(test)] <- pred[1:h,]
  test
}

# the simple model 3
# linear regression of time series 
tslm.basic <- function(train, test){
  # Computes a forecast using linear regression and seasonal dummy variables
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  horizon <- nrow(test)
  train[is.na(train)] <- 0
  # one train is for one dept 
  # one column is for one store
  for(j in 2:ncol(train)){
    # one store is one time series
    s <- ts(train[, j], frequency=52)
    model <- tslm(s ~ trend + season)
    fc <- forecast(model, h=horizon)
    test[, j] <- as.numeric(fc$mean)
  }
  test
}

# This is model 6 from the post.
###########
### the average of 3 simple models after postprocess
### mean(simple_model -> postprocess)
###########
simple.names <- c('tslm.basic', 'seasonal.naive', 'product')
shifts <- c(2.5, 2, 2)
simple.nums <- 0 * shifts

for(k in 1:3){
  print(paste('Predicting on model:', simple.names[k]))
  pred <- grouped.forecast(train, test, simple.names[k])
  print(paste('Shifting predictions for model:', simple.names[k]))
  pred <- postprocess(train, pred, shift=shifts[k])
  simple.nums[k] <- write.submission(pred)
}

pred <- make.average(simple.nums)
print('This is the shifted average of simple models.')

# Make the 3 simple models, shift their values and average them.
# The shifted average gets 2503 on the final board.



######################
### other models
### https://github.com/davidthaler/Walmart_competition_code/blob/master/grouped.forecast.R
######################

# This is model 3 from the post
# the one that averages predictions.

# Standard scaling + stlf/ets + averaging

# Instead, the data were standard scaled, 
# and a correlation matrix was computed. 
# Then forecasts were made and 
# several of the closely correlated series were averaged together, 
# before restoring the original scale.

stlf.nn <- function(train, test, method='ets', k, level1, level2){
  # Function standard scales the series and computes a correlation matrix.
  # Then it forecasts each store using stlf() from the forecast package.
  # That function performs an STL decomposition on each series, seasonally
  # adjusts the data, non-seasonally forecasts the seasonally adjusted data,
  # and then adds in the naively extended seasonal component to get the
  # final forecast.
  # Finally, it averages together some of the most correlated series before
  # restoring the original scale.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  # method - one of 'ets' or 'arima', specifies which type of model to
  #        use for the non-seasonal forecast
  # level1 - all series correlated to this level are used in the average
  # level2 - no series are used if they are correlated to less than this level
  # k - up to k series that are above level2 will be selected
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  horizon <- nrow(test)
  # delete date col, only cols of stores
  tr <- train[, 2:ncol(train)]
  tr[is.na(tr)] <- 0
  # the correlation between stores
  crl <- cor(tr)
  # Standard scaling
  tr.scale <- scale(tr)
  tr.scale[is.na(tr.scale)] <- 0
  raw.pred <- test[, 2:ncol(test)]
  # for every store
  for(j in 1:ncol(tr)){
    # transform scaled one store into time series
    s <- ts(tr.scale[, j], frequency=52)
    if(method == 'ets'){
      fc <- stlf(s, 
                 h=horizon, 
                 s.window=3, 
                 method='ets',
                 ic='bic', 
                 opt.crit='mae')
    }else if(method == 'arima'){
      fc <- stlf(s, 
                 h=horizon, 
                 s.window=3, 
                 method='arima',
                 ic='bic')
    }
    raw.pred[, j] <- fc$mean
  }
  # for every store
  for(j in 1:ncol(tr)){
    # sort the correlation between one store(such as 1) and others
    # return the most correlated store number
    o <- order(crl[j, ], decreasing=TRUE)
    # return the most correlated value
    score <- sort(crl[j, ], decreasing=TRUE)
    # if there are more than 5(k) store which correlated value > 0.95(level1)
    if(length(o[score >= level1]) > k){
      top.idx <- o[score >= level1]
    }else{
      top.idx <- o[score >= level2]
      top.idx <- top.idx[1:min(length(top.idx),k)]
    }
    # get top correlated prediction
    top <- raw.pred[, top.idx]
    if (length(top.idx) > 1){
      # mean(one day all high correlated store)
      pred <- rowMeans(top)
    }else{
      pred <- as.numeric(top)
    }
    # pred * the scale of one store 
    pred <- pred * attr(tr.scale, 'scaled:scale')[j]
    pred <- pred + attr(tr.scale, 'scaled:center')[j]
    test[, j + 1] <- pred
  }
  test
}

pred <- grouped.forecast(train, test, 'stlf.nn', k=5, level1=0.95, level2=0.8)
pred <- postprocess(train, pred, shift=2.5)


# This is model 4, the seasonal arima model.

# SVD + seasonal arima

# This used auto.arima() from the forecast package. 
# These models were actually all (p, d, q)(0, 1, 0)[52], 
# essentially non-seasonal arima errors on a seasonal naive model.

seasonal.arima.svd <- function(train, test, n.comp){
  # Replaces the training data with a rank-reduced approximation of itself
  # and then produces seasonal arima forecasts for each store.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  # n.comp - the number of components to keep in the singular value
  #         decomposition that is performed for preprocessing
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  horizon <- nrow(test)
  tr <- preprocess.svd(train, n.comp)
  # for every store
  for(j in 2:ncol(tr)){
    # if na value more than 1/3 nrow(train)
    # run fallback
    # else run arima
    if(sum(is.na(train[, j])) > nrow(train)/3){
      # Use DE model as fallback
      test[, j] <- fallback(tr[,j], horizon)
      store.num <- names(train)[j]
      print(paste('Fallback on store:', store.num))
    }else{
      # fit arima model
      s <- ts(tr[, j], frequency=52)
      model <- auto.arima(s, ic='bic', seasonal.test='ch')
      fc <- forecast(model, h=horizon)
      test[, j] <- as.numeric(fc$mean)
    }
  }
  test
}

# which is used in function 'seasonal.arima.svd'
fallback <- function(train, horizon){
  # This method is a fallback forecasting method in the case that there are
  # enough NA's to possibly crash arima models. It takes one seasonal 
  # difference, forecasts with a level-only exponential model, and then
  # inverts the seasonal difference.
  # 
  # args:
  # train - a vector of training data for one store
  # horizon - the forecast horizon in weeks
  #
  # returns:
  #  a vector of forecast values
  s <- ts(train, frequency=52)
  s[is.na(s)] <- 0
  # ses: exponential smoothing forecasts
  fc <- ses(diff(s, 52), h=horizon)
  result <- diffinv(fc$mean, lag=52, xi=s[length(s) - 51:0])
  result[length(result) - horizon:1 + 1]
}

pred <- grouped.forecast(train, test, 'seasonal.arima.svd', n.comp=15)
pred <- postprocess(train, pred, shift=2)

# This is model 5 from the post, regression on Fourier series terms with 
# non-seasonal arima errors. This model scores poorly on its own, but 
# improves the average anyway. This model is shifted by 1 because its
# period is 365/7, not 52. It is also very smooth, so the shift actually
# makes no difference here anyway.
#
# NB: This model may take a couple of hours to run

fourier.arima <- function(train, test, k){
  # This model is a regression on k sin/cos pairs of Fourier series terms
  # with non-seasonal arima errors. The call to auto.arima() crashes on data
  # with too many missing values, or too many identical values, so this 
  # function falls back to another, more stable method in that case.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  # k - number of sin/cos pair to use
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  horizon <- nrow(test)
  # for every store
  for(j in 2:ncol(train)){
    # deal with too many missing values
    if(sum(is.na(train[, j])) > nrow(train)/3){
      test[, j] <- fallback(train[,j], horizon)
      print(paste('Fallback on store:', names(train)[j]))
    }else{
      # fit arima model
      s <- ts(train[, j], frequency=365/7)
      model <- auto.arima(s, xreg=fourier(s, k), ic='bic', seasonal=FALSE)
      fc <- forecast(model, h=horizon, xreg=fourierf(s, k, horizon))
      test[, j] <- as.numeric(fc$mean)
    }
  }
  test
}

pred <- grouped.forecast(train, test, 'fourier.arima', k=12)
pred <- postprocess(train, pred, shift=1)




