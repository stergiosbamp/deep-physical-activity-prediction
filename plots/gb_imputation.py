import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


not_imputed = pd.read_csv('../results/imputation/gb_no_imputation.csv', index_col=0)
imputed = pd.read_csv('../results/imputation/gb_imputation.csv', index_col=0)

# r2 scores
not_imputed_r2 = not_imputed.loc['r2'][0]
imputed_r2 = imputed.loc['r2'][0]

r2s = [not_imputed_r2, imputed_r2]

# median aes
not_imputed_median = not_imputed.loc['median_ae'][0]
imputed_median = imputed.loc['median_ae'][0]

median_aes = [not_imputed_median, imputed_median]

# maes
not_imputed_mae = not_imputed.loc['mae'][0]
imputed_mae = imputed.loc['mae'][0]

maes = [not_imputed_mae, imputed_mae]

fig, ax = plt.subplots(1, 3, figsize=(15, 10))
fig.suptitle('Gradient Boosting Regressor for using or not imputation of zeros')
ax[0].set(ylabel='r2')
ax[1].set(ylabel='median_ae')
ax[2].set(ylabel='mae')

x = ['Not imputed', 'Imputed']

sns.barplot(x=x, y=r2s, order=x, hue=x, ax=ax[0])
sns.barplot(x=x, y=median_aes, order=x, hue=x, ax=ax[1])
sns.barplot(x=x, y=maes, order=x, hue=x, ax=ax[2])
plt.show()
