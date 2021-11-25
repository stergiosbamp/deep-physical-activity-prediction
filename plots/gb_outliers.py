import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


no_outliers = pd.read_csv('../results/outliers/gb_no_outliers.csv', index_col=0)
with_outliers = pd.read_csv('../results/outliers/gb_with_outliers.csv', index_col=0)

# r2 scores
no_r2 = no_outliers.loc['r2'][0]
with_r2 = with_outliers.loc['r2'][0]

r2s = [no_r2, with_r2]

# median aes
no_median = no_outliers.loc['median_ae'][0]
with_median = with_outliers.loc['median_ae'][0]

median_aes = [no_median, with_median]

# maes
no_mae = no_outliers.loc['mae'][0]
with_mae = with_outliers.loc['mae'][0]

maes = [no_mae, with_mae]


fig, ax = plt.subplots(1, 3, figsize=(15, 10))
fig.suptitle('Gradient Boosting Regressor for using outlier removal qith quantile method (q=0.05)')
ax[0].set(ylabel='r2')
ax[1].set(ylabel='median_ae')
ax[2].set(ylabel='mae')

x = ['No outliers', 'With outliers']

sns.barplot(x=x, y=r2s, order=x, hue=x, ax=ax[0])
sns.barplot(x=x, y=median_aes, order=x, hue=x, ax=ax[1])
sns.barplot(x=x, y=maes, order=x, hue=x, ax=ax[2])
plt.show()
