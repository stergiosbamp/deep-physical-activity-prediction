"""
Script for evaluating the Machine Learning (pre-trained) models.

Examples:
    Evaluating using the pre-trained Ridge model
        $ python evaluate.py --pretrained-model models/ridge.pkl
"""

import argparse

from sklearn.preprocessing import MinMaxScaler

from src.model.ml.baseline import BaselineModel
from src.model.ml.evaluator import MLEvaluator
from src.preprocessing.dataset import DatasetBuilder


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--pretrained-model', help='The pickle path for the scikit-learn pre-trained model')

    args = parser.parse_args()

    model_path = args.pretrained_model
    regressor = BaselineModel.load_model(model_path)

    # Get the dataset to get the same train/test splits
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../../data/datasets/hourly/df-3x24-just-steps.pkl')

    dataset = ds_builder.create_dataset_steps_features()
    x_train, x_val, x_test, y_train, y_val, y_test = ds_builder.get_train_val_test(dataset=dataset)

    evaluator = MLEvaluator(regressor=regressor)

    target_scaler = MinMaxScaler()

    # fit on train
    target_scaler.fit(y_train.values.reshape(-1, 1))

    # transform all targets
    y_train = target_scaler.transform(y_train.values.reshape(-1, 1))
    y_val = target_scaler.transform(y_val.values.reshape(-1, 1))
    y_test = target_scaler.transform(y_test.values.reshape(-1, 1))

    y_pred = evaluator.inference(x_test)
    y_pred_train = evaluator.inference(x_train)
    y_pred_val = evaluator.inference(x_val)

    # inverse transformation for performance recording
    y_test = target_scaler.inverse_transform(y_test)
    y_val = target_scaler.inverse_transform(y_val)
    y_train = target_scaler.inverse_transform(y_train)

    scores_test = evaluator.evaluate(y_test, y_pred)
    scores_train = evaluator.evaluate(y_train, y_pred_train)
    scores_val = evaluator.evaluate(y_val, y_pred_val)

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)
