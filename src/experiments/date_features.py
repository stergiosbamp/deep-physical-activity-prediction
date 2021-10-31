import os
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.model.baseline import BaselineModel
from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_VARIATION_DATASETS


def only_steps_and_cyclic_features():
    gb_pipe = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(verbose=1, random_state=1))

    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join(BASE_PATH_VARIATION_DATASETS,
                                                       'df-3*24-imputed-no-outliers-steps-and-cyclic-features.pkl'))
    X_train, X_test, y_train, y_test = ds_builder.get_train_test()

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)
    baseline_ml.set_pipe(gb_pipe)

    # record
    results = baseline_ml.score()

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/features/gb_hourly_steps_and_cyclic_features.csv')


def only_steps_features():
    gb_pipe = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(verbose=1, random_state=1))

    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join(BASE_PATH_VARIATION_DATASETS,
                                                       'df-3*24-imputed-no-outliers-steps-features.pkl'))
    X_train, X_test, y_train, y_test = ds_builder.get_train_test()

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)
    baseline_ml.set_pipe(gb_pipe)

    # record
    results = baseline_ml.score()

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/features/gb_hourly_steps_features.csv')


if __name__ == '__main__':
    only_steps_and_cyclic_features()
    only_steps_features()
