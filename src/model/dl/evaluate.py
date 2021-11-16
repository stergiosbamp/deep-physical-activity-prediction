import os

import matplotlib.pyplot as plt
import torch
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from src.config.directory import BASE_PATH_DAILY_DATASETS
from src.model.dl.lstm import LSTMRegressor
from src.model.dl.cnn import CNNRegressor
from src.model.dl.mlp import MLPRegressor
from src.preprocessing.dataset import DatasetBuilder


class Evaluator:
    def __init__(self, x_test, y_test, model, checkpoint_path):
        self.x_test = x_test
        self.y_test = y_test
        self.model = model.load_from_checkpoint(checkpoint_path)
        self.y_pred = None

    def evaluate(self):
        # Important: set it to evaluation mode
        self.model.eval()

        # Inference from PyTorch model and get predictions as numpy array
        y_pred = self.model(torch.from_numpy(self.x_test).float())
        y_pred = y_pred.detach().numpy()

        self.y_pred = y_pred

        # Now use predictions with any metric from sklearn
        print("MAE", mean_absolute_error(self.y_test, self.y_pred))
        print("R2", r2_score(self.y_test, self.y_pred))
        print("MdAE", median_absolute_error(self.y_test, self.y_pred))
        print("MAPE", mean_absolute_percentage_error(self.y_test, self.y_pred))

    def plot(self, smooth=False):
        x_range = self.y_test.index
        if smooth:
            df_preds = pd.DataFrame(self.y_pred)
            df_trues = pd.DataFrame(self.y_test)
            self.y_pred = df_preds.rolling(200).mean().values
            self.y_test = df_trues.rolling(200).mean().values

        plt.plot(x_range, self.y_test, label='true')
        plt.plot(x_range, self.y_pred, label='pred')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Get the dataset to get the same train/test splits
    ds_builder = DatasetBuilder(n_in=3 * 24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../../data/datasets/no-offset/df-3x24-just-steps.pkl')

    dataset = ds_builder.create_dataset_steps_features()
    x_train, x_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    # Scale the test set
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    evaluator = Evaluator(x_test=x_test,
                          y_test=y_test,
                          model=CNNRegressor,
                          checkpoint_path='lightning_logs/version_44/checkpoints/CNN-v1.ckpt')
    evaluator.evaluate()
    evaluator.plot(smooth=True)
