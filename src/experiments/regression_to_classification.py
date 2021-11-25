import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from src.preprocessing.dataset import DatasetBuilder


def classification():
    pipe = make_pipeline(MinMaxScaler(), DecisionTreeClassifier(random_state=1))

    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../data/datasets/variations/df-3x24-clf-just-steps.pkl',
                                classification=True)

    dataset = ds_builder.create_dataset_all_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    df = pd.DataFrame.from_dict(report).transpose()
    df.to_csv('../../results/classification/Trees-clf.csv',
              float_format='%.3f')


if __name__ == '__main__':
    classification()
