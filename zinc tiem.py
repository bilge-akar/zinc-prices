import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib as mpl
import matplotlib.pyplot as plt   # data visualization
import seaborn as sns             # statistical data visualization


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

path = r'C:\Users\bilge\Downloads\zinc.csv.xlsx'

df = pd.read_excel(path)

df.rename(columns={'date': 'Date', 'LME_Zinc_Cash_Settlement': 'Zinc'}, inplace=True)

df_sorted = df.sort_values(by='Date', ascending=True)

x = df_sorted['Date'].astype(str)
y = pd.to_numeric(df_sorted['Zinc'], errors='coerce')

plt.plot(x[::-1], y)

plt.xticks(rotation=90)  
num_ticks = 10  
step = len(x) // num_ticks  
plt.xticks(range(0, len(x), step), x[::step])

plt.xlabel('Date')
plt.ylabel('Zinc')
plt.title('Zinc Prices over Time')

plt.tight_layout()

plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


# Multiplicative Decomposition 
multiplicative_decomposition = seasonal_decompose(df['Zinc'], model='multiplicative', period=1000)

# Additive Decomposition
additive_decomposition = seasonal_decompose(df['Zinc'], model='additive', period=1000)

# Plot
plt.rcParams.update({'figure.figsize': (16,12)})
multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



plt.show()



trend_add = additive_decomposition.trend
seasonal_add = additive_decomposition.seasonal
residual_add = additive_decomposition.resid

sum_components = trend_add + seasonal_add

plt.rcParams.update({'figure.figsize': (16, 12)})
plt.plot(df.index, sum_components, label='Trend + Seasonal')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Sum of Trend and Seasonal Components')
plt.legend()
plt.show()

trend_mul = multiplicative_decomposition.trend
seasonal_mul = multiplicative_decomposition.seasonal
residual_mul = multiplicative_decomposition.resid

sum_components = trend_mul * seasonal_mul

plt.rcParams.update({'figure.figsize': (16, 12)})
plt.plot(df.index, sum_components, label='Trend x Seasonal')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Multiplication of Trend and Seasonal Components')
plt.legend()
plt.show()


residuals_mul = residual_mul.dropna()
mse_mul = np.mean(residuals_mul ** 2)
print("Multiplicative Decomposition MSE:", mse_mul)

from statsmodels.stats.diagnostic import acorr_ljungbox

lbvalue_mul, pvalue_mul = acorr_ljungbox(residuals_mul)
print("Multiplicative Decomposition Ljung-Box p-value:", pvalue_mul)

residuals_add = residual_add.dropna()
mse_add = np.mean(residuals_add ** 2)
print("Additive Decomposition MSE:", mse_add)

lbvalue_add, pvalue_add = acorr_ljungbox(residuals_add)
print("Additive Decomposition Ljung-Box p-value:", pvalue_add)
