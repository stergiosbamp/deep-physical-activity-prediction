import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import SGDRegressor, Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.preprocessing.dataset import DatasetBuilder
from src.model.ml.evaluate import MLEvaluator


class BaselineModel:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, regressor):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.pipe = make_pipeline(MinMaxScaler(), regressor)

        self.evaluator = MLEvaluator(x_train, x_val, x_test, y_train, y_val, y_test, self.pipe)

    def train_model(self):
        self.pipe.fit(self.x_train, self.y_train)

    def tune_model(self, X, y, grid_params):
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(self.pipe, grid_params, scoring='neg_median_absolute_error', cv=tscv, n_jobs=-1)
        grid.fit(X, y)

        self.pipe = grid.best_estimator_
        self.evaluator.regressor = self.pipe

        print("Best tuned pipeline: {}".format(self.pipe))

    def set_pipe(self, new_pipe):
        self.pipe = new_pipe


if __name__ == '__main__':
    dataset_builder = DatasetBuilder(n_in=3*24,
                                     granularity='whatever',
                                     save_dataset=True,
                                     directory='../../../data/datasets/variations/df-3x24-no-zeros-just-steps.pkl',
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

    scores_train = baseline_ml.evaluator.evaluate_train()
    scores_val = baseline_ml.evaluator.evaluate_val()
    scores_test = baseline_ml.evaluator.evaluate_test()

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)

    baseline_ml.evaluator.plot_predictions_train(smooth=True)
    baseline_ml.evaluator.plot_predictions(smooth=True)
