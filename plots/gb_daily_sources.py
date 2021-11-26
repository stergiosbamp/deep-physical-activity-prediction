import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


no_daily = pd.read_csv('../results/daily-sources/gb_without_daily_sources.csv', index_col=0)
with_daily = pd.read_csv('../results/daily-sources/gb_with_daily_sources.csv', index_col=0)

# r2 scores
no_daily_r2 = no_daily.loc['r2'][0]
with_daily_r2 = with_daily.loc['r2'][0]

r2s = [no_daily_r2, with_daily_r2]

# median aes
no_daily_median = no_daily.loc['median_ae'][0]
with_daily_median = with_daily.loc['median_ae'][0]

median_aes = [no_daily_median, with_daily_median]

# maes
no_daily_mae = no_daily.loc['mae'][0]
with_daily_mae = with_daily.loc['mae'][0]

maes = [no_daily_mae, with_daily_mae]

fig, ax = plt.subplots(1, 3, figsize=(15, 10))
fig.suptitle('Gradient Boosting Regressor for including or not the daily source identifiers')
ax[0].set(ylabel='r2')
ax[1].set(ylabel='median_ae')
ax[2].set(ylabel='mae')

x = ['Without daily', 'With daily']

sns.barplot(x=x, y=r2s, order=x, hue=x, ax=ax[0])
sns.barplot(x=x, y=median_aes, order=x, hue=x, ax=ax[1])
sns.barplot(x=x, y=maes, order=x, hue=x, ax=ax[2])
plt.show()
