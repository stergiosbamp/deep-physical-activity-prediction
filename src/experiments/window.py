"""
Module for running the experiment about investigating the impact
of daily vs hourly in conjunction with using 1 to 6 days before as window size.
"""

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.model.ml.evaluator import MLEvaluator
from src.preprocessing.dataset import DatasetBuilder
from src.model.ml.baseline import BaselineModel


HOURLY_WINDOWS = [
    1*24, 2*24, 3*24, 4*24, 5*24, 6*24
]

HOURLY_DATASET_PATHS = [
    '../../data/datasets/hourly/df-1x24-just-steps.pkl',
    '../../data/datasets/hourly/df-2x24-just-steps.pkl',
    '../../data/datasets/hourly/df-3x24-just-steps.pkl',
    '../../data/datasets/hourly/df-4x24-just-steps.pkl',
    '../../data/datasets/hourly/df-5x24-just-steps.pkl',
    '../../data/datasets/hourly/df-6x24-just-steps.pkl',
]

DAILY_WINDOWS = [
    1, 2, 3, 4, 5, 6
]

DAILY_DATASET_PATHS = [
    '../../data/datasets/daily/df-1-day-just-steps.pkl',
    '../../data/datasets/daily/df-2-day-just-steps.pkl',
    '../../data/datasets/daily/df-3-day-just-steps.pkl',
    '../../data/datasets/daily/df-4-day-just-steps.pkl',
    '../../data/datasets/daily/df-5-day-just-steps.pkl',
    '../../data/datasets/daily/df-6-day-just-steps.pkl',
]


def record_performance(regressor, windows, dataset_paths, path_results):

    # if os.path.exists(path_results):
    #     print("Experiment has already at {}".format(path_results))
    #     return

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

        baseline_ml = BaselineModel(regressor=regressor)
        model = baseline_ml.train_model(X_train, y_train)

        evaluator = MLEvaluator(model)
        y_pred = evaluator.inference(X_test)
        result = evaluator.evaluate(y_test, y_pred)

        # record
        results[window] = result

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(path_results)


if __name__ == '__main__':
    # Linear model
    ridge = Ridge(random_state=1)

    record_performance(ridge, HOURLY_WINDOWS, HOURLY_DATASET_PATHS,
                       '../../results/window/ridge_hourly_windows.csv')
    record_performance(ridge, DAILY_WINDOWS, DAILY_DATASET_PATHS,
                       '../../results/window/ridge_daily_windows.csv')
    # Tree model
    trees = DecisionTreeRegressor(random_state=1)
    record_performance(trees, HOURLY_WINDOWS, HOURLY_DATASET_PATHS,
                       '../../results/window/tree_hourly_windows.csv')
    record_performance(trees, DAILY_WINDOWS, DAILY_DATASET_PATHS,
                       '../../results/window/tree_daily_windows.csv')

    # Ensemble model
    gb = GradientBoostingRegressor(verbose=1, random_state=1)

    record_performance(gb, HOURLY_WINDOWS, HOURLY_DATASET_PATHS,
                       '../../results/window/gb_hourly_windows.csv')
    record_performance(gb, DAILY_WINDOWS, DAILY_DATASET_PATHS,
                       '../../results/window/gb_daily_windows.csv')
