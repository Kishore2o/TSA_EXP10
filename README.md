## Name : Kishore S
## Reg No : 212222240050
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the Netflix stock price dataset
data = pd.read_csv('/content/NFLX.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(data['Open'], color='blue')
plt.xlabel('Date')
plt.ylabel('Opening Stock Price')
plt.title('Netflix Stock Opening Price Time Series')
plt.show()

# Check stationarity of 'Open' prices
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['Open'])

# Seasonal Differencing for Potentially Better Stationarity
data['Open_diff'] = data['Open'] - data['Open'].shift(12)  # Assuming monthly seasonality
data.dropna(inplace=True)

# Plotting ACF and PACF
plot_acf(data['Open_diff'], lags=40)
plt.show()

plot_pacf(data['Open_diff'], lags=40)
plt.show()

# Manually defining ARIMA parameters (example: p=1, d=1, q=1, P=1, D=1, Q=1, m=12)
p, d, q = 1, 1, 1  # Non-seasonal ARIMA parameters
P, D, Q, m = 1, 1, 1, 12  # Seasonal ARIMA parameters

# Step 2: Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['Open'][:train_size], data['Open'][train_size:]

# Step 3: Fit the SARIMA model with manually chosen parameters
sarima_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
sarima_result = sarima_model.fit(disp=False)

# Step 4: Forecasting
predictions = sarima_result.predict(start=len(train), end=len(data)-1, dynamic=False)

# Step 5: Evaluate the model with RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Step 6: Plot Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Opening Stock Price')
plt.title('SARIMA Model Predictions for Netflix Stock Price')
plt.legend()
plt.show()

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/e8429503-d0bd-4eed-b28f-2ac03c889e1b)

![image](https://github.com/user-attachments/assets/cf2f28a2-48a7-4303-88d0-bf778b77272c)

![image](https://github.com/user-attachments/assets/805229c9-376a-4e1a-95c6-edccf0cafcf1)

![380474386-af2fe46a-2b75-4504-b697-1ddd320f1cc1](https://github.com/user-attachments/assets/8fe4f2f8-7c9c-4bc1-9bb9-c12fe5c06b62)

![image](https://github.com/user-attachments/assets/fb02f583-59dc-4775-8494-0b236dd9b40e)

### RESULT:
Thus, the SARIMA model was successfully implemented, accurately forecasting Google stock prices with measurable error evaluation.
