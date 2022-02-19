import joblib

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


class BaselineModel:
    """
    Class that provides training and tuning ML regression models for time-series problems.
    """

    def __init__(self, regressor):
        self.model = make_pipeline(MinMaxScaler(), regressor)

    def train_model(self, x, y):
        """
        Function that trains the model with features scaling.

        Args:
            x (np.array): Matrix of shape (n_samples, n_features).
            y (np.array): The target columns of shape (n_samples).

        Returns:
            (sklearn.Pipeline): The trained pipeline model.
        """

        self.model.fit(x, y)
        return self.model

    def tune_model(self, x, y, grid_params, path_to_save=None):
        """
        Function that tunes a time-series based model respecting chronological order for CV.

        Args:
            x (np.array): Matrix of shape (n_samples, n_features).
            y (np.array): The target columns of shape (n_samples).
            grid_params (dict): The dictionary holding the parameters to search for.
            path_to_save (str): RegressorMixin.

        Returns:
            (sklearn.Pipeline): The trained pipeline model.
        """

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
        """
        Function that sets a new scikit-learn regression model.

        Args:
            new_model (sklearn.base.RegressorMixin): An sklearn-style model.
        """

        self.model = new_model

    def save_model(self, path):
        """
        Function that saves a model.

        Args:
            path (str): The path to save the model.
        """

        joblib.dump(self.model, path)

    @staticmethod
    def load_model(path):
        """
        Function that loads a pre-trained model.

        Args:
            path (str): The path to load the pre-trained model.

        Returns:
            (sklearn.base.RegressorMixin): The loaded model.
        """

        model = joblib.load(path)
        return model
