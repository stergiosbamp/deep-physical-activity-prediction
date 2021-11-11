"""
Plot to study the optimal window from 1 to 6 days before, using the results
for the median ae and r2, from the gradient boosting regression.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df_hourly = pd.read_csv('../results/hourly-daily-chopped/ridge_hourly_windows.csv', index_col=0)
df_daily = pd.read_csv('../results/no-offset/ridge_daily_windows.csv', index_col=0)

# Set the window as 1, 2, .., 6 days instead of hours to be grouped under the same column
df_hourly['window'] = df_daily.index
df_daily['window'] = df_daily.index

# For separation
df_hourly['granularity'] = 'hourly'
df_daily['granularity'] = 'daily'

# Join them together, for plotting
df = pd.concat([df_daily, df_hourly])

fig, ax = plt.subplots(1, 2, figsize=(15, 10))

fig.suptitle('Ridge Regressor')
sns.barplot(data=df, x='window', y='median_ae', hue='granularity', ax=ax[0])
sns.barplot(data=df, x='window', y='r2', hue='granularity', ax=ax[1])

plt.show()
