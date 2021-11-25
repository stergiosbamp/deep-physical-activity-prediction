import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


all_features = pd.read_csv('../results/features/ridge_all_features.csv', index_col=0)
steps_cyclic_only = pd.read_csv('../results/features/ridge_steps_cyclic.csv', index_col=0)
steps_only = pd.read_csv('../results/features/ridge_steps_only.csv', index_col=0)

# r2 scores
both_dates_cyclic_r2 = all_features.loc['r2'][0]
only_cyclic_r2 = steps_cyclic_only.loc['r2'][0]
steps_only_r2 = steps_only.loc['r2'][0]

# median_ae scores
both_dates_cyclic_median = all_features.loc['median_ae'][0]
only_cyclic_median = steps_cyclic_only.loc['median_ae'][0]
steps_only_median = steps_only.loc['median_ae'][0]

r2s = [both_dates_cyclic_r2, only_cyclic_r2, steps_only_r2]
median_aes = [both_dates_cyclic_median, only_cyclic_median, steps_only_median]

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle('Gradient Boosting Regressor for using all/some/none date features')
ax[0].set(ylabel='r2')
ax[1].set(ylabel='median_ae')

x = ['both dates and cyclic', 'only cyclic', 'none (just steps)']

sns.barplot(x=x, y=r2s, order=x, hue=x, ax=ax[0])
sns.barplot(x=x, y=median_aes, order=x, hue=x, ax=ax[1])
plt.show()
