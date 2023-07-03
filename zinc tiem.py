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

path = r'Downloads\zinc.csv.xlsx'

df = pd.read_excel(path)

df.rename(columns={'date': 'Date', 'LME_Zinc_Cash_Settlement': 'Zinc'}, inplace=True)

x = df['Date'].astype(str)
y = pd.to_numeric(df['Zinc'], errors='coerce')

plt.plot(x[:-1], y)

plt.xticks(rotation=90)  # Rotate x-axis tick labels by 90 degrees

# Reduce the number of ticks to avoid overcrowding
num_ticks = 10  # Specify the desired number of ticks
step = len(x) // num_ticks  # Determine the step size
plt.xticks(range(0, len(x), step), x[::step])

plt.xlabel('Date')
plt.ylabel('Zinc')
plt.title('Zinc Prices over Time')

plt.tight_layout()

plt.show()

