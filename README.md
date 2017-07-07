# kaggle

I heard of kaggle 5 years ago, but several weeks ago I just activated my kaggle account, because one of my colleague proposed to do kaggle competitions together.

Below competitions are benefical to me and enlarging my horizon, especially the process of coding and thingking and study winner's code. 

I should do competitions earlier, so *delay* is actually a bad habit.


- **[Competition Series 1: Forecast use of a city bike share system](http://nbviewer.jupyter.org/github/yishi/kaggle/blob/master/competition_series_1_bike.ipynb)**

Data come from [kaggle](https://www.kaggle.com/c/bike-sharing-demand).
This is my first kaggle competition.

Firstly, I add features *year month hour* and used ensemble algorithms of extra trees regressor, the value of root mean squared logarithmic error in test is *0.47448*.

On top of this, I renew features with *rush_hour_working*, because the register users might go to work by bike and focus on rush hour such as 7:00 8:00 17:00 18:00 19:00; I also add feature *adverse_rush_hour* to descript the behavior of casual users who only have one peak from 9:00 to 20:00, the value of root mean squared logarithmic error in test decrease from 0.47448 to *0.44763*.


- **[Competition Series 2: Walmart store sales forecasting](https://github.com/yishi/kaggle/blob/master/competition_series_2_walmart.R)**

Data come from [kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting).

This time, I study the code of the first entry, which is mainly about time series model, such as expotential smoothing or arima, but he also use the simple model such as make the data of last year as predictor, the simple model have unexpected good effect, which give me a surprise and clue about how to simulate experience of specialist into a model.

In addition, the preprocess of singular value decomposition and the postprocess about shift the sales number around Chrismas are beneficial to me.

2017-7-7

