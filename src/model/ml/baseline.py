import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import SGDRegressor, Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.preprocessing.dataset import DatasetBuilder


class BaselineModel:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, regressor):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.pipe = make_pipeline(MinMaxScaler(), regressor)

        self.y_pred = None
        self.y_pred_train = None
        self.y_pred_val = None

        self.scores_test = dict()
        self.scores_train = dict()
        self.scores_val = dict()

    def evaluate_test(self):
        self.y_pred = self.pipe.predict(self.x_test)

        self.scores_test['R2'] = r2_score(self.y_test, self.y_pred)
        self.scores_test['MAE'] = mean_absolute_error(self.y_test, self.y_pred)
        self.scores_test['MAPE'] = mean_absolute_percentage_error(self.y_test, self.y_pred)
        self.scores_test['MdAE'] = median_absolute_error(self.y_test, self.y_pred)
        self.scores_test['RMSE'] = mean_squared_error(self.y_test, self.y_pred, squared=False)

        return self.scores_test

    def evaluate_train(self):
        self.y_pred_train = self.pipe.predict(self.x_train)

        self.scores_train['R2'] = r2_score(self.y_train, self.y_pred_train)
        self.scores_train['MAE'] = mean_absolute_error(self.y_train, self.y_pred_train)
        self.scores_train['MAPE'] = mean_absolute_percentage_error(self.y_train, self.y_pred_train)
        self.scores_train['MdAE'] = median_absolute_error(self.y_train, self.y_pred_train)
        self.scores_train['RMSE'] = mean_squared_error(self.y_train, self.y_pred_train, squared=False)

        return self.scores_train

    def evaluate_val(self):
        self.y_pred_val = self.pipe.predict(self.x_val)

        # Now use predictions with any metric from sklearn
        self.scores_val['R2'] = r2_score(self.y_val, self.y_pred_val)
        self.scores_val['MAE'] = mean_absolute_error(self.y_val, self.y_pred_val)
        self.scores_val['MAPE'] = mean_absolute_percentage_error(self.y_val, self.y_pred_val)
        self.scores_val['MdAE'] = median_absolute_error(self.y_val, self.y_pred_val)
        self.scores_val['RMSE'] = mean_squared_error(self.y_val, self.y_pred_val, squared=False)

        return self.scores_val

    def train_model(self):
        self.pipe.fit(self.x_train, self.y_train)

    def tune_model(self, X, y, grid_params):
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(self.pipe, grid_params, scoring='neg_median_absolute_error', cv=tscv, n_jobs=-1)
        grid.fit(X, y)
        self.pipe = grid.best_estimator_
        print("Best tuned pipeline: {}".format(self.pipe))

    def plot_predictions(self, smooth=False):
        x_range = self.y_test.index
        if smooth:
            df_preds = pd.DataFrame(self.y_pred)
            df_trues = pd.DataFrame(self.y_test)
            self.y_pred = df_preds.rolling(200).mean().values
            self.y_test = df_trues.rolling(200).mean().values
        plt.plot(x_range, self.y_test, label='true')
        plt.plot(x_range, self.y_pred, label='pred')
        plt.legend()
        plt.show()

    def plot_predictions_train(self, smooth=False):
        x_range = self.y_train.index
        if smooth:
            df_preds = pd.DataFrame(self.y_pred_train)
            df_trues = pd.DataFrame(self.y_train)
            self.y_pred_train = df_preds.rolling(200).mean().values
            self.y_train = df_trues.rolling(200).mean().values
        plt.plot(x_range, self.y_train, label='true')
        plt.plot(x_range, self.y_pred_train, label='pred')
        plt.legend()
        plt.show()

    def set_pipe(self, new_pipe):
        self.pipe = new_pipe


if __name__ == '__main__':
    dataset_builder = DatasetBuilder(n_in=3*24,
                                     granularity='whatever',
                                     save_dataset=True,
                                     directory='../../../data/datasets/variations/df-3x24-no-wear-days-500-just-steps.pkl',
                                     total_users=None)

    dataset = dataset_builder.create_dataset_all_features()
    x_train_val, _, y_train_val, _ = dataset_builder.get_train_test(dataset=dataset)
    x_train, x_val, x_test, y_train, y_val, y_test = dataset_builder.get_train_val_test(dataset=dataset)

    grid = {
        "ridge__alpha": [1, 5, 10, 15, 20]
    }

    regressor = Ridge(random_state=1)

    baseline_ml = BaselineModel(x_train=x_train, x_val=x_val, x_test=x_test,
                                y_train=y_train, y_val=y_val, y_test=y_test,
                                regressor=regressor)

    baseline_ml.tune_model(X=x_train_val, y=y_train_val, grid_params=grid)

    scores_train = baseline_ml.evaluate_train()
    scores_val = baseline_ml.evaluate_val()
    scores_test = baseline_ml.evaluate_test()

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)

    baseline_ml.plot_predictions_train(smooth=True)
    baseline_ml.plot_predictions(smooth=True)
