import os
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.model.ml.baseline import BaselineModel
from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_HOURLY_DATASETS


def all_features():
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-all-features.pkl')
    dataset = ds_builder.create_dataset_all_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)

    # record
    results = baseline_ml.score()

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/features/ridge_all_features.csv')


def only_steps_and_cyclic_features():
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-all-features.pkl')
    dataset = ds_builder.create_dataset_steps_cyclic_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)

    # record
    results = baseline_ml.score()

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

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)

    # record
    results = baseline_ml.score()

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/features/ridge_steps_only.csv')


if __name__ == '__main__':
    all_features()
    only_steps_and_cyclic_features()
    only_steps_features()
