import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


not_imputed = pd.read_csv('../results/gb_hourly_not_imputed_no_outliers.csv', index_col=0)
imputed = pd.read_csv('../results/gb_hourly_imputed_no_outliers.csv', index_col=0)

# r2 scores
not_imputed_r2 = not_imputed.loc['r2'][0]
imputed_r2 = imputed.loc['r2'][0]

r2s = [not_imputed_r2, imputed_r2]

# median aes
not_imputed_median = not_imputed.loc['median_ae'][0]
imputed_median = imputed.loc['median_ae'][0]

median_aes = [not_imputed_median, imputed_median]

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle('Gradient Boosting Regressor for using or not imputation of zeros')
ax[0].set(ylabel='r2')
ax[1].set(ylabel='median_ae')

x = ['Not imputed', 'Imputed']

sns.barplot(x=x, y=r2s, order=x, hue=x, ax=ax[0])
sns.barplot(x=x, y=median_aes, order=x, hue=x, ax=ax[1])
plt.show()
