import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


upper_outliers = pd.read_csv('../results/outliers/gb_hourly_imputed_upper_outliers.csv', index_col=0)
both_outliers = pd.read_csv('../results/outliers/gb_hourly_imputed_both_outliers.csv', index_col=0)
with_outliers = pd.read_csv('../results/outliers/gb_hourly_imputed_with_outliers.csv', index_col=0)

# r2 scores
upper_r2 = upper_outliers.loc['r2'][0]
both_r2 = both_outliers.loc['r2'][0]
with_r2 = with_outliers.loc['r2'][0]

r2s = [upper_r2, both_r2, with_r2]

# median aes
upper_median = upper_outliers.loc['median_ae'][0]
both_median = both_outliers.loc['median_ae'][0]
with_median = with_outliers.loc['median_ae'][0]

median_aes = [upper_median, both_median, with_median]

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle('Gradient Boosting Regressor for using outlier removal qith quantile method (q=0.05)')
ax[0].set(ylabel='r2')
ax[1].set(ylabel='median_ae')

x = ['Upper quantile only', 'Upper & lower quantiles', 'With outliers']

sns.barplot(x=x, y=r2s, order=x, hue=x, ax=ax[0])
sns.barplot(x=x, y=median_aes, order=x, hue=x, ax=ax[1])
plt.show()
