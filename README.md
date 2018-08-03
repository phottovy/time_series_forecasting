# time_series_forecasting

<!-- # 14,0,14,1 -->

## Background
For this capstone, I am working with an outside company that works heavily with time series data. They currently utilize six "ARIMA-like" prediction methods and I have been tasked with analyzing and (hopefully) optimizing these methods. They have also given me the freedom to come up with my own forecasting methods if in the hopes of improving their daily forecasts. I will be working with this company for my final capstone project as well, so this is part one of a two step process.

### Data
The dataset for this company consists of 10 months of daily customer activity to be used as my testing and training data. Along with the activity column, each date index has column with the forecast of day 0, the forecast for that date, through the next 14 days. There are also six unique tables for each of the six current prediction methods. The date indices and actual column are the same for each of the six tables. In addition, there are ten different files, or different feeds, with different variations of the customer activity metric. For the scope of the project, I am only going to focus on analyzing and forecasting for one of the feeds and then compare my findings with the rest of the feeds during part two of this project. Finally, each of the feeds comes with a separate data provenance file containing various metrics on the forecasts in the main file. My main focus for part one of this project is the time series activity but I will take a closer look at these metrics during the second phase of this project.

To summarize, unlike the rest of my colleagues in the DSI who are working with data rich in columns and features, part one of my capstone project primarily involves two columns, the date index and the true activity and then the corresponding forecasts for this daily activity.

## Process
>"I only need to make predictions using one feature, that doesn't sound too difficult!" -Me, foolishly thinking to myself, when I first started working with the dataset

Once I had the data, my first thought was to jump right in and start making predictions using the ARIMA method (more on this method later), which is what the current metrics are partially based on.

Step one: **Visualize the Data**
![raw][1]

Step two: **Start making predictions**
![first_attempt][2]

Even though my first forecast was _pretty close to perfect_ (italics = sarcasm), I decided I needed to learn more about timeseries data before I just plug numbers into a model and hope for the best. Below is a summary of what I discovered during my journey into the wonders of time series analytics.








[1]: images/raw_data.png
[2]: images/first_attempt.png
[3]: images/rolling_mean.png
[4]: images/activity_by_date_1.png
[5]: images/activity_by_date_2.png
[6]: images/activity_by_date_3.png
[7]: images/activity_by_date_4.png
[8]: images/decomp_plots.png
