import os
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_VARIATION_DATASETS


def all_features():
    pipe = make_pipeline(MinMaxScaler(), DecisionTreeClassifier(random_state=1))

    ds_builder = DatasetBuilder(n_in=3 * 24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join(BASE_PATH_VARIATION_DATASETS,
                                                       'df-3*24-classification.pkl'),
                                classification=True)

    X_train, X_test, y_train, y_test = ds_builder.get_train_test()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    df = pd.DataFrame.from_dict(report).transpose()
    df.to_csv('../../results/clf_decision_trees_hourly_classification_all_features.csv', float_format='%.3f')


def steps_cyclic():
    pipe = make_pipeline(MinMaxScaler(), DecisionTreeClassifier(random_state=1))

    ds_builder = DatasetBuilder(n_in=3 * 24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join(BASE_PATH_VARIATION_DATASETS,
                                                       'df-3*24-classification-steps-cyclic.pkl'),
                                classification=True)

    X_train, X_test, y_train, y_test = ds_builder.get_train_test()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    df = pd.DataFrame.from_dict(report).transpose()
    df.to_csv('../../results/clf_decision_trees_hourly_classification_steps_cyclic.csv', float_format='%.3f')


def steps_only():
    pipe = make_pipeline(MinMaxScaler(), DecisionTreeClassifier(random_state=1))

    ds_builder = DatasetBuilder(n_in=3 * 24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join(BASE_PATH_VARIATION_DATASETS,
                                                       'df-3*24-classification-steps-only.pkl'),
                                classification=True)

    X_train, X_test, y_train, y_test = ds_builder.get_train_test()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    df = pd.DataFrame.from_dict(report).transpose()
    df.to_csv('../../results/clf_decision_trees_hourly_classification_steps_only.csv', float_format='%.3f')


if __name__ == '__main__':
    all_features()
    steps_cyclic()
    steps_only()