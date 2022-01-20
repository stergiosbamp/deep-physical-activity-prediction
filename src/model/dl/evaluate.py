import argparse

from sklearn.preprocessing import MinMaxScaler

from src.model.dl.lstm import LSTMRegressor
from src.model.dl.cnn import CNNRegressor
from src.model.dl.mlp import MLPRegressor
from src.preprocessing.dataset import DatasetBuilder
from src.model.ml.evaluator import DLEvaluator


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ckpt_path', help='The checkpoint path for the PyTorch pre-trained model')
    parser.add_argument('--model', help='One of RNN | CNN | MLP')

    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    model_type = args.model

    if model_type == 'RNN':
        model = LSTMRegressor
    elif model_type == 'CNN':
        model = CNNRegressor
    elif model_type == 'MLP':
        model = MLPRegressor
    else:
        print("The model must be one of: RNN, CNN, MLP. Not {}", model_type)
        exit()

    # Get the dataset to get the same train/test splits
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../../data/datasets/hourly/df-3x24-just-steps.pkl')

    dataset = ds_builder.create_dataset_steps_features()
    x_train, x_val, x_test, y_train, y_val, y_test = ds_builder.get_train_val_test(dataset=dataset)

    # Scale the test set
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    evaluator = DLEvaluator(x_train, x_val, x_test, y_train, y_val, y_test,
                            model=model,
                            ckpt_path=ckpt_path)

    scores_train = evaluator.evaluate_train()
    scores_val = evaluator.evaluate_val()
    scores_test = evaluator.evaluate_test()

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)

    # evaluator.save_results(scores_train, "cnn-train.csv")
    # evaluator.save_results(scores_val, "cnn-val.csv")
    # evaluator.save_results(scores_test, "cnn-test.csv")

    # evaluator.plot_predictions_train(smooth=True)
    # evaluator.plot_predictions(smooth=True)
