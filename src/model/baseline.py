import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.dataset import DatasetBuilder


if __name__ == '__main__':
    dataset_builder = DatasetBuilder(n_in=5*24,
                                     granularity='1H',
                                     save_dataset=True,
                                     directory='../../data/df-5-day-imputed-no-outliers-all-features-all-users-with'
                                               '-subject-injected.pkl',
                                     total_users=None)

    X_train, X_test, y_train, y_test = dataset_builder.get_train_test()

    pipe = make_pipeline(MinMaxScaler(), Ridge(random_state=1))

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("r2", r2_score(y_test, y_pred))
    print("mae", mean_absolute_error(y_test, y_pred))
    print("mape", mean_absolute_percentage_error(y_test, y_pred))
    print("median ae", median_absolute_error(y_test, y_pred))

    y_test_val = y_test.values
    x_range = X_test.index
    plt.plot(x_range, y_test_val, label='true')
    plt.plot(x_range, y_pred, label='pred')
    plt.legend()
    plt.show()
