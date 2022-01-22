import joblib

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


class BaselineModel:
    def __init__(self, regressor):
        self.model = make_pipeline(MinMaxScaler(), regressor)

    def train_model(self, x, y):
        self.model.fit(x, y)
        return self.model

    def tune_model(self, x, y, grid_params, path_to_save=None):
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(self.model, grid_params, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1)
        grid.fit(x, y)

        print("Best parameters:", grid.best_params_)
        print("Best tuned pipeline:", grid.best_estimator_)

        self.model = grid.best_estimator_
        if path_to_save is not None:
            self.save_model(path_to_save)

        return self.model

    def set_model(self, new_model):
        self.model = new_model

    def save_model(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load_model(path):
        model = joblib.load(path)
        return model
