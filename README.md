# Time Series Forecasting


## Background
For this capstone, I am working with an outside company that works heavily with time series data. They currently utilize six "ARIMA-like" prediction methods and I have been tasked with analyzing and (hopefully) optimizing these methods. They have also given me the freedom to come up with my own forecasting methods if in the hopes of improving their daily forecasts. I will be working with this company for my final capstone project as well, so this is part one of a two step process.

#### Data
The dataset for this company consists of 10 months of daily customer activity to be used as my testing and training data. Along with the activity column, each date index has column with the forecast of day 0, the forecast for that date, through the next 14 days. There are also six unique tables for each of the six current prediction methods. The date indices and actual column are the same for each of the six tables. In addition, there are ten different files, or different feeds, with different variations of the customer activity metric. For the scope of the project, I am only going to focus on analyzing and forecasting for one of the feeds and then compare my findings with the rest of the feeds during part two of this project. Finally, each of the feeds comes with a separate data provenance file containing various metrics on the forecasts in the main file. My main focus for part one of this project is the time series activity but I will take a closer look at these metrics during the second phase of this project.

To summarize, unlike the rest of my colleagues in the DSI who are working with data rich in columns and features, part one of my capstone project primarily involves two columns, the date index and the true activity and then the corresponding forecasts for this daily activity.

## EDA

#### Initial Analysis
>"I only need to make predictions using one feature, that doesn't sound too difficult!" -Me, foolishly thinking to myself, when I first started working with the dataset

Once I had the data, my first thought was to jump right in and start making predictions using the ARIMA method (more on this method later), which is what the current metrics are partially based on.

Step one: **Visualize the Data**
![raw][1]

Step two: **Start making predictions**
![first_attempt][2]

Even though my first forecast was _pretty close to perfect_ (italics = sarcasm), I decided I needed to learn more about timeseries data before I just plug numbers into a model and hope for the best. Below is a summary of what I discovered during my journey into the wonders of time series analytics.

#### Visualize and Evaluate
One of the main things I discovered during this project is just how much you can learn about time series data from simple visualization. The shape and patterns of the raw data have significant impacts on how well different forecasting methods will perform with the data.

A good first step is add a rolling mean and standard deviation to your plot to get a clearer view of the different trends in your data.

![rolling][3]

Even though the data has a steady pattern of ups and downs, the rolling mean and std provide a smoothing view of the data to see where it is trending.

Next, I tried to see if there were any patterns in the increases and decreases. Since I have daily values, I decided to look for trends within days of the weeks.

![q1][4]
![q2][5]
![q3][6]
![q4][7]

As you can see, there is a very clear drop off in activity on the weekends.This basic discovery shows some consistency in the data that will help with forecasting.

#### Initial Analytics
Two common approaches to find patterns in the data are Time Series Decomposition and Testing for Stationarity:

Using a **decomposition plot**, you can decompose the data into three components:
 * Trend
 * Seasonality
 * Noise

![decomp][8]

With our data, there is a lot of movement in the trend, but it is clear that there is consistency in seasonality and a relatively flat residual plot.

With time series data, having a stationary trend allows for better forecasting since we can presuppose that future values will be consistent with current values. One approach to explore the stationarity of the data is by using an **Augmented Dickey-Fuller test (ADF)**. This use a hypothesis testing approach to for stationarity with the null hypothesis stating that the series is **not stationary** and the alternative hypothesis that it is stationary. Using statsmodels built in `adfuller` function, our date gets the following results:

![adf][9]
\* We will discuss the correlation plots later.

 **Augmented Dickey-Fuller Full Results:**
 * T-Statistic: -3.00941
 * p-value: 0.03402
 * adf-1%: -3.45310
 * adf-5%: -2.87156
 * adf-10%: -2.57211
 * \# Lags: 14
 * \# of observations: 290
 * 0.03402 <= 0.05. Data **is stationary**

Fortunately our data is stationary which means we can start forecasting! There are techniques to convert non-stationary data to be stationary but I am not going to go into details about those.

## Forecast
As stated before, this is a journey into time series analysis so we are going to start at the bottom and work our way up.

#### Moving Average Forecast
The most basic approach is to use the rolling mean we used earlier to forecast future values.

Using the \# of lags calculated in the ADF test, I used a 14 day rolling mean for my predictions

![move_avg][10]

Clearly the predictions will follow the overall trend of the series but it will give you the desired results.

#### Exponential Smoothing
One of the next techniques is exponential smoothing which has multiple levels:

 **Single Exponential Smoothing**
![exp_1][11]

 **Double Exponential Smoothing**
![exp_2][12]

 **Triple exponential smoothing a.k.a. Holt-Winters**
![exp_3][13]


## ARIMA Forecasting
Stands for **Autoregressive Integrated Moving Average (ARIMA)**

Optimized to adjust for lags. 14 was the best number.

![adf_lags][14]
![arima][15]
![arima_diag][16]


## Analysis of Original Predictors
- differenced_1_data     81 days
- differenced_1_cycle    76 days
- cycle_series           74 days
- data                   55 days
- differenced_2_data      3 days
- differenced_2_cycle     1 days

## How did my predictors do?
- differenced_1_data     80 days
- differenced_1_cycle    74 days
- cycle_series           73 days
- data                   55 days
- mov_avg                 4 days
- differenced_2_data      3 days
- differenced_2_cycle     1 days



## Next Steps
 * Finish readme
 * Boosting
 * RNN & LSTM




References
End to end
topic 1 part 9
clickfox
jose portilla
interpretting error






[1]: images/raw_data.png
[2]: images/first_attempt.png
[3]: images/rolling_mean.png
[4]: images/activity_by_date_1.png
[5]: images/activity_by_date_2.png
[6]: images/activity_by_date_3.png
[7]: images/activity_by_date_4.png
[8]: images/decomp_plots.png
[9]: images/adf_plot.png
[10]: images/rolling_mean_forecast.png
[11]: images/exponential_smooth.png
[12]: images/doub_exp_smooth.png
[13]: images/trip_exp_smooth.png
[14]: images/adf_with_lags.png
[15]: images/arima_forecast.png
[16]: images/arima_diagnostics.png
