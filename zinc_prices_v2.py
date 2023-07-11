import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib as mpl
import matplotlib.pyplot as plt   # data visualization
import seaborn as sns             # statistical data visualization

from datetime import datetime
import numpy as np             #for numerical computations like log,exp,sqrt etc
import pandas as pd            #for reading & storing data, pre-processing
import matplotlib.pylab as plt #for visualization
            
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10, 6

file = 'zinc.csv.xlsx'

# Reading the data from excel and plotting a zinc price-time graph

df = pd.read_excel(file)
df.dropna(inplace=True)
df.shape



df.rename(columns={'date': 'Date', 'LME_Zinc_Cash_Settlement': 'Zinc'}, inplace=True)

x = df['Date'].astype(str)
y = pd.to_numeric(df['Zinc'], errors='coerce') 

plt.plot(x, y)

plt.xticks(rotation=90)  
num_ticks = 10  
step = len(x) // num_ticks  
plt.xticks(range(0, len(x), step), x[::step])

plt.xlabel('Date')
plt.ylabel('Zinc')
plt.title('Zinc Prices over Time')

plt.tight_layout()

plt.show()

df.head()
df.tail()

# Test for stationarity

from statsmodels.tsa.stattools import adfuller

def stationarity(timeseries: pd.DataFrame, column):
    
    rolmean=timeseries.rolling(365).mean()
    rolstd=timeseries.rolling(365).std()
    
    plt.figure(figsize=(20,10))
    actual=plt.plot(timeseries[column], color='red', label='Actual')
    mean_6=plt.plot(rolmean, color='green', label='Rolling Mean') 
    std_6=plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    print('Dickey-Fuller Test: ')
    dftest=adfuller(timeseries[column], autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

stationarity(df, 'Zinc')



# Calculate the log of the adjusted close prices
df_zinc_log = np.log(df['Zinc'])


# Plot the transformed series
plt.figure(figsize=(10, 6))
plt.plot(df_zinc_log, color='blue', label='Transformed Series')
plt.xlabel('Time')
plt.ylabel('Zinc')
plt.title('Log Transformed Series')
plt.legend()
plt.show()

df_log = pd.DataFrame()
df_log["Zinc"] = df_zinc_log

stationarity(df_log, 'Zinc')

MAvg=df_log.rolling(window=12).mean()
MStd=df_log.rolling(window=12).std()

df_log_diff=df_log-MAvg

df_log_diff=df_log_diff.dropna()

# Plot the transformed series
plt.figure(figsize=(10, 6))
plt.plot(df_log_diff, color='blue', label='Transformed Series')
plt.xlabel('Time')
plt.ylabel('Zinc')
plt.title('Log-Diff Transformed Series')
plt.legend()
plt.show()

stationarity(df_log_diff,'Zinc')


# Test for seasonality
from pandas.plotting import autocorrelation_plot

# Draw Plot
plt.rcParams.update({'figure.figsize':(10,5), 'figure.dpi':120})
autocorrelation_plot(df['Zinc'].tolist())
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


# Multiplicative Decomposition 
multiplicative_decomposition = seasonal_decompose(df['Zinc'], model='multiplicative', period=365)

# Additive Decomposition
additive_decomposition = seasonal_decompose(df['Zinc'], model='additive', period=365)

# Plot
plt.rcParams.update({'figure.figsize': (5,10)})
multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.show()






#ARIMA


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df_log_diff['Zinc'], order=(2, 1, 3))
results = model.fit()

plt.figure(figsize=(20, 10))
plt.plot(df_log_diff)
plt.plot(results.fittedvalues, color='red')
plt.title('RSS')
plt.show()

predictions = pd.Series(results.fittedvalues, copy=True)

predictions_log = pd.Series(df_log_diff['Zinc'].iloc[0], index=df_log_diff.index)

predictions_cum_sum = predictions.cumsum()
predictions_log_added = predictions_log.add(predictions_cum_sum, fill_value=0)
predictions_ARIMA = np.exp(predictions_log_added)

# Plot the transformed series
plt.figure(figsize=(20, 10))
plt.plot(df['Zinc'])
plt.plot(predictions_ARIMA)
plt.xlabel('Time')
plt.ylabel('Zinc')
plt.title('Actual vs. Predicted')
plt.legend(['Actual', 'Predicted'])
plt.show()

last_day = df.index[-1]

forecast_steps = 365
forecast = results.get_forecast(steps=forecast_steps)

forecast_days = np.arange(last_day + 1, last_day + forecast_steps + 1)

forecast_values = np.exp(forecast.predicted_mean)

forecast_diff = pd.Series(forecast_values, index=forecast_days)
forecast_log_added = pd.Series(df_log['Zinc'].iloc[-1], index=forecast_days).add(forecast_diff.cumsum(), fill_value=0)
forecast_ARIMA = np.exp(forecast_log_added)

# Plot the forecasted values
plt.figure(figsize=(20, 10))
plt.plot(df['Zinc'], label='Actual')
plt.plot(forecast_ARIMA, color='red', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Zinc')
plt.title('Zinc Prices - Actual vs. Forecast')
plt.legend()
plt.show()

# Print the forecasted values
print("Forecasted Values:")
print(forecast_ARIMA)
