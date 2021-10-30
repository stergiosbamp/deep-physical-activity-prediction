import os
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.model.baseline import BaselineModel
from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_VARIATION_DATASETS


def removing_upper_outliers():
    pipe = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(verbose=1, random_state=1))

    ds_builder = DatasetBuilder(n_in=3 * 24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join(BASE_PATH_VARIATION_DATASETS,
                                                       'df-3*24-imputed-upper-outliers.pkl'))

    X_train, X_test, y_train, y_test = ds_builder.get_train_test()

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)
    baseline_ml.set_pipe(pipe)

    # record
    results = baseline_ml.score()
    print(results)

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/gb_hourly_imputed_upper_outliers.csv')

    # plot
    baseline_ml.plot_predictions()


def removing_all_outliers():
    pipe = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(verbose=1, random_state=1))

    ds_builder = DatasetBuilder(n_in=3 * 24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join(BASE_PATH_VARIATION_DATASETS,
                                                       'df-3*24-imputed-both-outliers.pkl'))

    X_train, X_test, y_train, y_test = ds_builder.get_train_test()

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)
    baseline_ml.set_pipe(pipe)

    # record
    results = baseline_ml.score()
    print(results)

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/gb_hourly_imputed_both_outliers.csv')

    # plot
    baseline_ml.plot_predictions()


def keep_outliers():
    pipe = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(verbose=1, random_state=1))

    ds_builder = DatasetBuilder(n_in=3 * 24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join(BASE_PATH_VARIATION_DATASETS,
                                                       'df-3*24-imputed-with-outliers.pkl'))

    X_train, X_test, y_train, y_test = ds_builder.get_train_test()

    baseline_ml = BaselineModel(X_train, X_test, y_train, y_test)
    baseline_ml.set_pipe(pipe)

    # record
    results = baseline_ml.score()
    print(results)

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/gb_hourly_imputed_with_outliers.csv')

    # plot
    baseline_ml.plot_predictions()


if __name__ == '__main__':
    removing_upper_outliers()
    removing_all_outliers()
    keep_outliers()
