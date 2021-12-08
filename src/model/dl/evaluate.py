from sklearn.preprocessing import MinMaxScaler

from src.model.dl.lstm import LSTMRegressor
from src.model.dl.cnn import CNNRegressor
from src.model.dl.mlp import MLPRegressor
from src.preprocessing.dataset import DatasetBuilder
from src.model.ml.evaluate import DLEvaluator


if __name__ == '__main__':
    # Get the dataset to get the same train/test splits
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../../data/datasets/variations/df-3x24-no-zeros-just-steps.pkl')

    dataset = ds_builder.create_dataset_steps_features()
    x_train, x_val, x_test, y_train, y_val, y_test = ds_builder.get_train_val_test(dataset=dataset)

    # Scale the test set
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    evaluator = DLEvaluator(x_train, x_val, x_test, y_train, y_val, y_test,
                            model=LSTMRegressor,
                            ckpt_path='lightning_logs/version_10/checkpoints/LSTM-batch-64-epoch-100-hidden-100-layers-2-dropout-0.2-lr-0.05.ckpt')

    scores_train = evaluator.evaluate_train()
    scores_val = evaluator.evaluate_val()
    scores_test = evaluator.evaluate_test()

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)

    evaluator.plot_predictions_train(smooth=True)
    evaluator.plot_predictions(smooth=True)
