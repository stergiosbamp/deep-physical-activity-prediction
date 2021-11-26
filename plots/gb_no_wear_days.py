import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


dropped = pd.read_csv('../results/no-wear-days/gb_drop.csv', index_col=0)
keep = pd.read_csv('../results/no-wear-days/gb_keep.csv', index_col=0)

# r2 scores
dropped_r2 = dropped.loc['r2'][0]
keep_r2 = keep.loc['r2'][0]

r2s = [dropped_r2, keep_r2]

# median aes
dropped_median = dropped.loc['median_ae'][0]
keep_median = keep.loc['median_ae'][0]

median_aes = [dropped_median, keep_median]

# maes
dropped_mae = dropped.loc['mae'][0]
keep_mae = keep.loc['mae'][0]

maes = [dropped_mae, keep_mae]

fig, ax = plt.subplots(1, 3, figsize=(15, 10))
fig.suptitle('Gradient Boosting Regressor for removing or keeping no wear days (< 500 steps)')
ax[0].set(ylabel='r2')
ax[1].set(ylabel='median_ae')
ax[2].set(ylabel='mae')

x = ['Remove no wear days', 'Keep no wear days']

sns.barplot(x=x, y=r2s, order=x, hue=x, ax=ax[0])
sns.barplot(x=x, y=median_aes, order=x, hue=x, ax=ax[1])
sns.barplot(x=x, y=maes, order=x, hue=x, ax=ax[2])
plt.show()
