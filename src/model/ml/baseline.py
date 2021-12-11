from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

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
        grid = GridSearchCV(self.pipe, grid_params, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1)
        grid.fit(X, y)

        print("Best parameters:", grid.best_params_)
        print("Best tuned pipeline:", grid.best_estimator_)

        self.pipe = grid.best_estimator_
        self.evaluator.regressor = self.pipe

    def set_pipe(self, new_pipe):
        self.pipe = new_pipe
