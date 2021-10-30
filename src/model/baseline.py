import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_HOURLY_DATASETS, BASE_PATH_DAILY_DATASETS


class BaselineModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.pipe = make_pipeline(MinMaxScaler(), Ridge(random_state=1))
        self.y_pred = None
        self.res = dict()

    def score(self):
        self.pipe.fit(self.x_train, self.y_train)
        self.y_pred = self.pipe.predict(self.x_test)

        self.res['r2'] = r2_score(self.y_test, self.y_pred)
        self.res['mae'] = mean_absolute_error(self.y_test, self.y_pred)
        self.res['mape'] = mean_absolute_percentage_error(self.y_test, self.y_pred)
        self.res['median_ae'] = median_absolute_error(self.y_test, self.y_pred)

        return self.res

    def plot_predictions(self):
        x_range = self.x_test.index
        plt.plot(x_range, self.y_test.values, label='true')
        plt.plot(x_range, self.y_pred, label='pred')
        plt.legend()
        plt.show()

    def set_pipe(self, new_pipe):
        self.pipe = new_pipe


if __name__ == '__main__':
    dataset_builder = DatasetBuilder(n_in=5*24,
                                     granularity='1H',
                                     save_dataset=True,
                                     directory='../../data/datasets/daily/df-3-day-imputed-no-outliers-all-features'
                                               '-all-users-with-subject-injected.pkl',
                                     total_users=None)

    X_train, X_test, y_train, y_test = dataset_builder.get_train_test()

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)

    scores = baseline_ml.score()
    print(scores)

    baseline_ml.plot_predictions()
