import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error, mean_squared_error


class BaseEvaluator:
    """
    Base class that evaluates a ML model.
    """

    def __init__(self):
        self.results_folder = Path("../../results/modeling")

    def inference(self, data):
        """
        Abstract method that makes a prediction

        Args:
            data (np.array): The data to predict.

        Returns:
            (np.array): The predicted data.
        """

        raise NotImplemented("Abstract method")

    def evaluate(self, true, pred):
        """
        Function to evaluate a regression problem based on
            - MAE
            - MdAE
            - R2 score
            - RMSE

        Args:
            true (np.array): The true values.
            pred (np.array): The predicted values.

        Returns:
            (dict): A dict that holds the calculated metrics.
        """

        scores = dict()

        scores['R2'] = r2_score(true, pred)
        scores['MAE'] = mean_absolute_error(true, pred)
        scores['MdAE'] = median_absolute_error(true, pred)
        scores['RMSE'] = mean_squared_error(true, pred, squared=False)

        return scores

    @staticmethod
    def plot(true, pred):
        """
        Function that plots the lines for true VS predictied values to estimate the time-series forecasting.

        Args:
            true (np.array): The true values.
            pred (np.array): The predicted values.

        """

        end = true.shape[0]
        x_range = np.arange(0, end)

        plt.plot(x_range, true, label='true')
        plt.plot(x_range, pred, label='pred')
        plt.legend()
        plt.show()

    def save_results(self, scores, filename):
        """
        Function that saves the metrics as a csv file.

        Args:
            scores (dict): Dictinary holding the metrics.
            filename (str): The path to save the csv file.
        """

        dest_path = Path(self.results_folder, filename)
        dest_path.with_suffix(".csv")

        if dest_path.exists():
            print("Modeling results for {} has already run".format(filename))
            return

        df = pd.DataFrame.from_dict(scores, orient='index')
        df.to_csv(dest_path.__str__())
        print("Saved results modeling for", filename)


class MLEvaluator(BaseEvaluator):
    """
    Evaluator for ML models for inferencing on sklearn style.
    """

    def __init__(self, regressor):
        super().__init__()
        self.regressor = regressor

    def inference(self, data):
        return self.regressor.predict(data)


class DLEvaluator(BaseEvaluator):
    """
    Evaluator for DL models for inferencing on PyTorch style as array
    for using any sklearn metric.
    """

    def __init__(self, model, ckpt_path):
        super().__init__()
        self.model = model.load_from_checkpoint(ckpt_path)
        self.model.freeze()

    def inference(self, data):
        tensor_preds = self.model(torch.from_numpy(data).float())
        return tensor_preds.numpy()
