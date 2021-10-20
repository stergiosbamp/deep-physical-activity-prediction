"""
Module that uses the Ridge regressor, in order to find and record the performance
from the hourly and daily window datasets.

Note that these datasets do not have outlier values removed. However by-hand experiments showed that this does not
affect much the performance.
"""

import os
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.dataset import DatasetBuilder


HOURLY_WINDOWS = [
    1*24, 2*24, 3*24, 4*24, 5*24
]

HOURLY_DATASET_PATHS = [
    '../../data/df-1*24-all-features-all-users-with-subject-injected.pkl',
    '../../data/df-2*24-all-features-all-users-with-subject-injected.pkl',
    '../../data/df-3*24-all-features-all-users-with-subject-injected.pkl',
    '../../data/df-4*24-all-features-all-users-with-subject-injected.pkl',
    '../../data/df-5*24-all-features-all-users-with-subject-injected.pkl'
]

DAILY_WINDOWS = [
    1, 2, 3, 4, 5
]

DAILY_DATASET_PATHS = [
    '../../data/df-1-day-all-features-all-users-with-subject-injected.pkl',
    '../../data/df-2-day-all-features-all-users-with-subject-injected.pkl',
    '../../data/df-3-day-all-features-all-users-with-subject-injected.pkl',
    '../../data/df-4-day-all-features-all-users-with-subject-injected.pkl',
    '../../data/df-5-day-all-features-all-users-with-subject-injected.pkl'
]


def record_performance(model, windows, dataset_paths, path_results):

    if os.path.exists(path_results):
        return

    results = {}

    for window, dataset_path in zip(windows, dataset_paths):
        # create dataset
        dataset_builder = DatasetBuilder(n_in=window,
                                         granularity='whatever',
                                         save_dataset=True,
                                         directory=dataset_path,
                                         total_users=None)

        X_train, X_test, y_train, y_test = dataset_builder.get_train_test()

        # fit/predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # scoring
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        median_ae = median_absolute_error(y_test, y_pred)

        # record
        results[window] = {"r2": r2, "mae": mae, "mape": mape, "median_ae": median_ae}

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(path_results)


if __name__ == '__main__':
    # Linear model
    ridge_pipe = make_pipeline(MinMaxScaler(), Ridge(random_state=1))

    record_performance(ridge_pipe, HOURLY_WINDOWS, HOURLY_DATASET_PATHS,
                       '../../results/ridge_hourly_windows_performance.csv')
    record_performance(ridge_pipe, DAILY_WINDOWS, DAILY_DATASET_PATHS,
                       '../../results/ridge_daily_windows_performance.csv')

    # Tree model
    trees_pipe = make_pipeline(MinMaxScaler(), DecisionTreeRegressor(random_state=1))
    record_performance(trees_pipe, HOURLY_WINDOWS, HOURLY_DATASET_PATHS,
                       '../../results/tree_hourly_windows_performance.csv')
    record_performance(trees_pipe, DAILY_WINDOWS, DAILY_DATASET_PATHS,
                       '../../results/tree_daily_windows_performance.csv')

    # Ensemble model
