"""
Module for running the experiment about investigating the impact
of imputing zero value activity in the active hours.
"""

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from src.model.ml.baseline import BaselineModel
from src.model.ml.evaluator import MLEvaluator
from src.preprocessing.dataset import DatasetBuilder


def no_imputation():
    gb_regressor = GradientBoostingRegressor(verbose=1, random_state=1)

    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/hourly/df-3x24-just-steps.pkl')

    dataset = ds_builder.create_dataset_all_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(regressor=gb_regressor)
    model = baseline_ml.train_model(X_train, y_train)

    evaluator = MLEvaluator(model)
    y_pred = evaluator.inference(X_test)
    results = evaluator.evaluate(y_test, y_pred)
    print(results)

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/imputation/gb_no_imputation.csv')


def with_imputation():
    gb_regressor = GradientBoostingRegressor(verbose=1, random_state=1)

    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-imputed-just-steps.pkl')

    dataset = ds_builder.create_dataset_all_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(regressor=gb_regressor)
    model = baseline_ml.train_model(X_train, y_train)

    evaluator = MLEvaluator(model)
    y_pred = evaluator.inference(X_test)
    results = evaluator.evaluate(y_test, y_pred)
    print(results)

    # # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/imputation/gb_imputation.csv')


if __name__ == '__main__':
    no_imputation()
    with_imputation()
