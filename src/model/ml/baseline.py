import joblib

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.model.ml.evaluator import MLEvaluator


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

    def tune_model(self, X, y, grid_params, path_to_save=None):
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(self.pipe, grid_params, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1)
        grid.fit(X, y)

        print("Best parameters:", grid.best_params_)
        print("Best tuned pipeline:", grid.best_estimator_)

        self.pipe = grid.best_estimator_
        if path_to_save is not None:
            self.save_model(path_to_save)

        self.evaluator.regressor = self.pipe

    def set_pipe(self, new_pipe):
        self.pipe = new_pipe

    def save_model(self, path):
        joblib.dump(self.pipe, path)

    @staticmethod
    def load_model(path):
        model = joblib.load(path)
        return model
