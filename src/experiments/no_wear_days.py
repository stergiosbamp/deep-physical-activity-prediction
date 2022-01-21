import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.model.ml.baseline import BaselineModel
from src.preprocessing.dataset import DatasetBuilder


def keep_no_wear_days():
    gb_regressor = GradientBoostingRegressor(verbose=1, random_state=1)

    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/hourly/df-3x24-just-steps.pkl')

    dataset = ds_builder.create_dataset_all_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(X_train, None, X_test, y_train, None, y_test, gb_regressor)
    baseline_ml.evaluator.zero_preds = False

    # record
    baseline_ml.train_model()
    results = baseline_ml.evaluator.evaluate_test()

    # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/no-wear-days/gb_keep.csv')


def drop_no_wear_days():
    gb_regressor = GradientBoostingRegressor(verbose=1, random_state=1)

    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-no-wear-days-500-just-steps.pkl')

    dataset = ds_builder.create_dataset_all_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    baseline_ml = BaselineModel(X_train, None, X_test, y_train, None, y_test, gb_regressor)
    baseline_ml.evaluator.zero_preds = False

    # record
    baseline_ml.train_model()
    results = baseline_ml.evaluator.evaluate_test()

    # # write them to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('../../results/no-wear-days/gb_drop.csv')


if __name__ == '__main__':
    keep_no_wear_days()
    drop_no_wear_days()
