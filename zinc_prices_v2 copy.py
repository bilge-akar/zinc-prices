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

df_log_diff=df_log.diff()

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


df_diff=df.diff()

df_diff=df_diff.dropna()

# Plot the transformed series
plt.figure(figsize=(10, 6))
plt.plot(df_diff['Zinc'], color='blue', label='Transformed Series')
plt.xlabel('Time')
plt.ylabel('Zinc')
plt.title('Diff Transformed Series')
plt.legend()
plt.show()

stationarity(df_diff,'Zinc')


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

# Fit the ARIMA model
model = ARIMA(df_log_diff, order=(2, 1, 3))
results = model.fit()

# Plot the fitted values
plt.figure(figsize=(10, 6))
plt.plot(df_log_diff)
plt.plot(results.fittedvalues, color='red')
plt.title('Residuals')
plt.show()

# Obtain the predicted values in the original scale
predictions_log = pd.Series(df_log.iloc[0], index=df_log.index)
predictions_cum_sum = results.fittedvalues.cumsum()
predictions_log_added = predictions_log.add(predictions_cum_sum, fill_value=0)
predictions_ARIMA = np.exp(predictions_log_added)

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(df['Zinc'], label='Actual')
plt.plot(predictions_ARIMA, color='red', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Zinc')
plt.title('Actual vs. Predicted')
plt.legend()
plt.show()

# Generate forecasted values
forecast_steps = 120
index_of_fc = np.arange(len(df), len(df)+forecast_steps)
fc = results.forecast(120, alpha=0.05)
fc_log = pd.Series(df_log['Zinc'], index=df_log.index)
fc_cum_sum = fc.cumsum()
fc_log_added = fc_log.add(fc_cum_sum, fill_value=0)
fc_ARIMA = np.exp(fc_log_added)

fc_series = pd.Series(fc_ARIMA, index=index_of_fc)


# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(df['Zinc'], label='Actual')
plt.plot(fc_series, color='red', label='Forecast')
plt.xlabel('Index')
plt.ylabel('Zinc')
plt.title('Actual vs. Forecast')
plt.legend()
plt.show()

# Print the forecasted values
print("Forecasted Values:")
print(fc_series)
print(fc)
print(len(df))