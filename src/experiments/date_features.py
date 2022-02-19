"""
Module for running the experiment about investigating the impact
of date features engineering. Specifically it evaluates the dataset
with steps only features, the dataset with steps and date features and the
dataset with steps values and the cyclic transformation of the date features (cyclic features).
"""

import pandas as pd

from sklearn.linear_model import Ridge
from src.model.ml.baseline import BaselineModel
from src.model.ml.evaluator import MLEvaluator
from src.preprocessing.dataset import DatasetBuilder


def all_features():
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-all-features.pkl')
    dataset = ds_builder.create_dataset_all_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(regressor=regressor)
    model = baseline_ml.train_model(X_train, y_train)

    evaluator = MLEvaluator(model)
    y_pred = evaluator.inference(X_test)
    results = evaluator.evaluate(y_test, y_pred)
    print(results)

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/features/ridge_all_features.csv')


def only_steps_and_date_features():
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-all-features.pkl')
    dataset = ds_builder.create_dataset_steps_date_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(regressor=regressor)
    model = baseline_ml.train_model(X_train, y_train)

    evaluator = MLEvaluator(model)
    y_pred = evaluator.inference(X_test)
    results = evaluator.evaluate(y_test, y_pred)
    print(results)

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/features/ridge_steps_date.csv')


def only_steps_and_cyclic_features():
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-all-features.pkl')
    dataset = ds_builder.create_dataset_steps_cyclic_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(regressor=regressor)
    model = baseline_ml.train_model(X_train, y_train)

    evaluator = MLEvaluator(model)
    y_pred = evaluator.inference(X_test)
    results = evaluator.evaluate(y_test, y_pred)
    print(results)

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/features/ridge_steps_cyclic.csv')


def only_steps_features():
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-all-features.pkl')
    dataset = ds_builder.create_dataset_steps_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(regressor=regressor)
    model = baseline_ml.train_model(X_train, y_train)

    evaluator = MLEvaluator(model)
    y_pred = evaluator.inference(X_test)
    results = evaluator.evaluate(y_test, y_pred)
    print(results)

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/features/ridge_steps_only.csv')


if __name__ == '__main__':
    regressor = Ridge(random_state=1)

    all_features()
    only_steps_and_date_features()
    only_steps_and_cyclic_features()
    only_steps_features()
