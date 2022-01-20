import argparse

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

    evaluator = MLEvaluator(x_train, x_val, x_test, y_train, y_val, y_test, regressor)

    scores_train = evaluator.evaluate_train()
    scores_val = evaluator.evaluate_val()
    scores_test = evaluator.evaluate_test()

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)
