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
"""
#ACF & PACF plots

lag_acf = acf(df_log_diff, nlags=20)
lag_pacf = pacf(df_log_diff, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')            

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout()

plt.show()
"""

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df['Zinc'].tolist(), lags=50, ax=axes[0])
plot_pacf(df['Zinc'].tolist(), lags=50, ax=axes[1])

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df_log['Zinc'].tolist(), lags=50, ax=axes[0])
plot_pacf(df_log['Zinc'].tolist(), lags=50, ax=axes[1])

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df_log_diff['Zinc'].tolist(), lags=50, ax=axes[0])
plot_pacf(df_log_diff['Zinc'].tolist(), lags=50, ax=axes[1])

plt.show()

from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(df['Zinc'], lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Zinc Prices', y=1.05)    
plt.show()

'''
from statsmodels.tsa.arima_model import ARIMA

plt.figure(figsize=(20,10))
model=ARIMA(df_log_diff, order=(2,1,2))
results=model.fit(disp=-1)
plt.plot(data_shift) ????????????????
plt.plot(results.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results.fittedvalues-data_shift['Passengers'])**2))
print('plotting ARIMA model')
'''