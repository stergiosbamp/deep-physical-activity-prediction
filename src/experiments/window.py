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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.dataset import DatasetBuilder
from src.model.baseline import BaselineModel
from src.config.directory import BASE_PATH_DAILY_DATASETS, BASE_PATH_HOURLY_DATASETS


HOURLY_WINDOWS = [
    1*24, 2*24, 3*24, 4*24, 5*24, 6*24
]

HOURLY_DATASET_PATHS = [
    os.path.join(BASE_PATH_HOURLY_DATASETS,
                 'df-1*24-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_HOURLY_DATASETS,
                 'df-2*24-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_HOURLY_DATASETS,
                 'df-3*24-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_HOURLY_DATASETS,
                 'df-4*24-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_HOURLY_DATASETS,
                 'df-5*24-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_HOURLY_DATASETS,
                 'df-6*24-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
]

DAILY_WINDOWS = [
    1, 2, 3, 4, 5, 6
]

DAILY_DATASET_PATHS = [
    os.path.join(BASE_PATH_DAILY_DATASETS,
                 'df-1-day-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_DAILY_DATASETS,
                 'df-2-day-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_DAILY_DATASETS,
                 'df-3-day-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_DAILY_DATASETS,
                 'df-4-day-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_DAILY_DATASETS,
                 'df-5-day-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
    os.path.join(BASE_PATH_DAILY_DATASETS,
                 'df-6-day-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'),
]


def record_performance(pipe, windows, dataset_paths, path_results):

    if os.path.exists(path_results):
        print("Experiment has already at {}".format(path_results))
        return

    results = {}

    for window, dataset_path in zip(windows, dataset_paths):
        # create dataset
        dataset_builder = DatasetBuilder(n_in=window,
                                         granularity='whatever',
                                         save_dataset=True,
                                         directory=dataset_path,
                                         total_users=None)

        dataset = dataset_builder.create_dataset_all_features()
        X_train, X_test, y_train, y_test = dataset_builder.get_train_test(dataset=dataset)

        baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)
        baseline_ml.set_pipe(pipe)

        # record
        results[window] = baseline_ml.score()

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(path_results)


if __name__ == '__main__':
    # Linear model
    ridge_pipe = make_pipeline(MinMaxScaler(), Ridge(random_state=1))

    record_performance(ridge_pipe, HOURLY_WINDOWS, HOURLY_DATASET_PATHS,
                       '../../results/window/ridge_hourly_windows_performance-imputed-no-outliers.csv')
    record_performance(ridge_pipe, DAILY_WINDOWS, DAILY_DATASET_PATHS,
                       '../../results/window/ridge_daily_windows_performance-imputed-no-outliers.csv')

    # Tree model
    trees_pipe = make_pipeline(MinMaxScaler(), DecisionTreeRegressor(random_state=1))
    record_performance(trees_pipe, HOURLY_WINDOWS, HOURLY_DATASET_PATHS,
                       '../../results/window/tree_hourly_windows_performance-imputed-no-outliers.csv')
    record_performance(trees_pipe, DAILY_WINDOWS, DAILY_DATASET_PATHS,
                       '../../results/window/tree_daily_windows_performance-imputed-no-outliers.csv')

    # Ensemble model
    gb_pipe = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(verbose=1, random_state=1))

    record_performance(gb_pipe, HOURLY_WINDOWS, HOURLY_DATASET_PATHS,
                       '../../results/window/gb_hourly_windows_performance-imputed-no-outliers.csv')
    record_performance(gb_pipe, DAILY_WINDOWS, DAILY_DATASET_PATHS,
                       '../../results/window/gb_daily_windows_performance-imputed-no-outliers.csv')
