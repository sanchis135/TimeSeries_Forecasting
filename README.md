# PROBLEM: Crude Oil Prices (TIME SERIES - FORECASTING)

## Objective
The main goal is to predict future Crude Oil Prices based on the historical data available in the dataset.

I compared the results using different forecasting models such as LSTM, ARIMA, SARIMA and Prophet. And I selected the best between them.

## Dataset
The crude oil price movements are subject to diverse influencing factors. This dataset was retrieved from the U.S. Energy Information Administration: Europe Brent Spot Price FOB (Dollars per Barrel)

The data contains daily Brent oil prices from 17th of May 1987 until the 13th of November 2022.

You can download this dataset in this link: https://www.kaggle.com/datasets/mabusalah/brent-oil-prices

Example of some rows of the datset:

![alt text](images/image1.png)


## EDA
Visualization of the full data:

![alt text](images/image2.png)

Seasonality Check:

The Augmented Dickey-Fuller (ADF) test is a statistical test used to determine whether a time series is stationary or has a unit root. In simpler terms, it helps to check if a time series data set is stable over time or if it shows trends, cycles, or other non-stationary behaviors.

The ADF test is an extension of the Dickey-Fuller test and includes more complex models to account for higher-order autoregressive processes. The test works by testing the null hypothesis that a unit root is present in the time series sample. If the null hypothesis is rejected, it suggests that the time series is stationary

------------------------------------------------------------------
We are working with the series Original Series

ADF Statistic Value from the precalculated tables: -2.89
ADF Statistic Value: -1.9918544071295257

Significance Level to consider the series as stationary: 0.05
p-value: 0.29015603926422134

Conclusion: The p-value is greater than the significance level of 0.05, so we fail to reject the null hypothesis. This means the Original Series is not stationary and has a unit root.

------------------------------------------------------------------
We are working with the series Log Series

ADF Statistic Value from the precalculated tables: -2.89
ADF Statistic Value: -1.8032940879348056

Significance Level to consider the series as stationary: 0.05
p-value: 0.37882750115650377

Conclusion: Similar to the Original Series, the p-value is greater than 0.05, so we fail to reject the null hypothesis. The Log Series is also not stationary and has a unit root.

------------------------------------------------------------------
We are working with the series Diff Log Series

ADF Statistic Value from the precalculated tables: -2.89
ADF Statistic Value: -16.427113494485894

Significance Level to consider the series as stationary: 0.05
p-value: 2.4985801611428892e-29

Conclusion: The p-value is significantly less than 0.05, so we reject the null hypothesis. This indicates that the Diff Log Series is stationary and does not have a unit root.

------------------------------------------------------------------

In summary, only the Diff Log Series is stationary, while the Original Series and Log Series are not.

## Models

ARIMA stands for Autoregressive Integrated Moving Average. It is a popular statistical method used for time series analysis and forecasting. 

Components of ARIMA:
- Autoregressive (AR) part: This component involves regressing the variable on its own lagged (past) values. The order of the AR part is denoted by 
p, which indicates the number of lagged values included in the model.
- Integrated (I) part: This component involves differencing the data to make it stationary. The order of differencing is denoted by 
d, which indicates the number of times the data has been differenced.
- Moving Average (MA) part: This component involves modeling the error term as a linear combination of error terms from previous time steps. The order of the MA part is denoted by q, which indicates the number of lagged forecast errors included in the model.

Notation
An ARIMA model is typically denoted as ARIMA(p,d,q), where:
- p is the number of lagged values in the autoregressive part.
- d is the number of times the data has been differenced.
- q is the number of lagged forecast errors in the moving average part.

Steps to Build an ARIMA Model
- Identification: Determine if the time series is stationary. If not, apply differencing to achieve stationarity. Identify the values of 
p,d, and q using autocorrelation (ACF) and partial autocorrelation (PACF) plots.
- Estimation: Estimate the parameters of the ARIMA model using the identified values of p, d, and q.
- Diagnostic Checking: Evaluate the model by checking the residuals to ensure they resemble white noise (i.e., they are uncorrelated and have a constant mean and variance).

For time series data with seasonality, the ARIMA model can be extended to include seasonal components. This is known as the SARIMA model, denoted as ARIMA(p,d,q)(P,D,Q)_m, where:
- P, D, and Q are the seasonal autoregressive, differencing, and moving average terms, respectively.
- m is the number of periods in each season.

ARIMA models are widely used in various fields such as finance, economics, and environmental science for forecasting future values based on past data. They are particularly useful for short-term forecasting when the underlying data shows patterns that can be captured by the model.

The basic ARIMA model can be extended further by incorporating the seasonality of the series and exogenous variables. In this case, we would be talking about the SARIMAX model represented by (p, d, q) x (P, D, Q) S: where the parameters (P, D, Q) represent the same idea as (p, d, q) but deal with the seasonal part of the series.

The parameter S, in turn, represents the number of periods that must pass for the seasonality to repeat: 12 for months, 4 for quarters, etc.

Next, we will use a basic "Gridsearch" to find the optimal parameters for the ARIMA model.

The AIC index is the Akaike Information Criterion and is used to choose a model from a set of possible models.

The index calculates the Kullback-Leibler distance between the model and the series.

One way to interpret the index is: we look for the model with the lowest index because it is the simplest one that fits the data.

Results:

![alt text](images/image3.png)

The best model is (0, 0, 2), with AIC = 22338.763866305933

Results of the best model:

![alt text](images/image4.png)


![alt text](images/image5.png)

Prediction:

![alt text](images/image6.png)

ARIMA model with the parameters (0, 0, 2) has as result of the rmse in test of 8.6

## Improvement

This resutls can be improved with other methods as Prophet applied on data serie.

Prophet is an open-source forecasting tool developed by Facebook's Core Data Science team. It is designed for time series forecasting and is particularly effective for data with strong seasonal patterns and historical data spanning multiple seasons.

Key Features:
- Additive Model: Prophet uses an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, along with holiday effects. This makes it suitable for time series data with complex seasonal patterns.
- Automatic Forecasting: Prophet is designed to be easy to use and fully automatic. It can generate forecasts with minimal manual intervention, making it accessible for both data scientists and analysts.
- Robustness: The model is robust to missing data and shifts in the trend, and it typically handles outliers well.
- Customizability: While Prophet provides automated forecasts, it also allows for manual adjustments and tuning to improve forecast accuracy.

Prophet is available in both R and Python, making it versatile for different programming environments. 

Prophet is widely used for various applications, including:
- Sales Forecasting: Predicting future sales based on historical data.
- Capacity Planning: Estimating future resource requirements.
- Financial Forecasting: Projecting future financial metrics.
- Demand Forecasting: Anticipating future demand for products or services.

Advantages:
- Ease of Use: Prophet is user-friendly and requires minimal coding to generate forecasts.
- Flexibility: It can handle different types of seasonality and incorporate holiday effects.
- Scalability: Suitable for large datasets and can be used for forecasting at scale.

Prophet is a powerful tool for time series forecasting, especially when dealing with seasonal data. 

Plot forecast:

![alt text](images/image7.png)

Plot individual components of forecast: trend, weekly/yearly seasonality:


![alt text](images/image8.png)
