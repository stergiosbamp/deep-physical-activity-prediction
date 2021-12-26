import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error, mean_squared_error


class BaseEvaluator:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, scaler, zero_preds=True):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.scaler = scaler

        self.zero_preds = zero_preds

        self.y_pred = None
        self.y_pred_train = None
        self.y_pred_val = None

        self.scores_test = dict()
        self.scores_train = dict()
        self.scores_val = dict()
        self.results_folder = Path("../../results/modeling")

    def inference(self, data):
        raise NotImplemented("Abstract method")

    def evaluate_test(self):
        self.y_pred = self.inference(self.x_test)

        self.y_pred = self.y_pred.numpy()

        zeros = np.zeros((self.y_pred.shape[0], 72))

        stacked_preds = np.hstack((zeros, self.y_pred.reshape(-1, 1)))
        inv_stacked = self.scaler.inverse_transform(stacked_preds)
        preds = inv_stacked[:, -1]

        self.y_pred = preds

        stacked_trues = np.hstack((zeros, self.y_test.reshape(-1, 1)))
        inv_stacked = self.scaler.inverse_transform(stacked_trues)
        trues = inv_stacked[:, -1]

        self.y_test = trues

        if self.zero_preds:
            self.y_pred = pd.DataFrame(self.y_pred)
            self.y_pred = self.y_pred[0].apply(lambda x: self._zero_prediction(x))

        self.scores_test['R2'] = r2_score(self.y_test, self.y_pred)
        self.scores_test['MAE'] = mean_absolute_error(self.y_test, self.y_pred)
        self.scores_test['MAPE'] = mean_absolute_percentage_error(self.y_test, self.y_pred)
        self.scores_test['MdAE'] = median_absolute_error(self.y_test, self.y_pred)
        self.scores_test['MSE'] = mean_squared_error(self.y_test, self.y_pred, squared=True)
        self.scores_test['RMSE'] = mean_squared_error(self.y_test, self.y_pred, squared=False)

        return self.scores_test

    def evaluate_train(self):
        self.y_pred_train = self.inference(self.x_train)

        self.y_pred_train = self.y_pred_train.numpy()

        zeros = np.zeros((self.y_pred_train.shape[0], 72))

        stacked_preds = np.hstack((zeros, self.y_pred_train.reshape(-1, 1)))
        inv_stacked = self.scaler.inverse_transform(stacked_preds)
        preds = inv_stacked[:, -1]

        self.y_pred_train = preds

        stacked_trues = np.hstack((zeros, self.y_train.reshape(-1, 1)))
        inv_stacked = self.scaler.inverse_transform(stacked_trues)
        trues = inv_stacked[:, -1]

        self.y_train = trues

        if self.zero_preds:
            self.y_pred_train = pd.DataFrame(self.y_pred_train)
            self.y_pred_train = self.y_pred_train[0].apply(lambda x: self._zero_prediction(x))

        self.scores_train['R2'] = r2_score(self.y_train, self.y_pred_train)
        self.scores_train['MAE'] = mean_absolute_error(self.y_train, self.y_pred_train)
        self.scores_train['MAPE'] = mean_absolute_percentage_error(self.y_train, self.y_pred_train)
        self.scores_train['MdAE'] = median_absolute_error(self.y_train, self.y_pred_train)
        self.scores_train['MSE'] = mean_squared_error(self.y_train, self.y_pred_train, squared=True)
        self.scores_train['RMSE'] = mean_squared_error(self.y_train, self.y_pred_train, squared=False)

        return self.scores_train

    def evaluate_val(self):
        self.y_pred_val = self.inference(self.x_val)

        self.y_pred_val = self.y_pred_val.numpy()

        zeros = np.zeros((self.y_pred_val.shape[0], 72))

        stacked_preds = np.hstack((zeros, self.y_pred_val.reshape(-1, 1)))
        inv_stacked = self.scaler.inverse_transform(stacked_preds)
        preds = inv_stacked[:, -1]

        self.y_pred_val = preds

        stacked_trues = np.hstack((zeros, self.y_val.reshape(-1, 1)))
        inv_stacked = self.scaler.inverse_transform(stacked_trues)
        trues = inv_stacked[:, -1]

        self.y_val = trues

        if self.zero_preds:
            self.y_pred_val = pd.DataFrame(self.y_pred_val)
            self.y_pred_val = self.y_pred_val[0].apply(lambda x: self._zero_prediction(x))

        # Now use predictions with any metric from sklearn
        self.scores_val['R2'] = r2_score(self.y_val, self.y_pred_val)
        self.scores_val['MAE'] = mean_absolute_error(self.y_val, self.y_pred_val)
        self.scores_val['MAPE'] = mean_absolute_percentage_error(self.y_val, self.y_pred_val)
        self.scores_val['MdAE'] = median_absolute_error(self.y_val, self.y_pred_val)
        self.scores_val['MSE'] = mean_squared_error(self.y_val, self.y_pred_val, squared=True)
        self.scores_val['RMSE'] = mean_squared_error(self.y_val, self.y_pred_val, squared=False)

        return self.scores_val

    def plot_predictions(self, smooth=False):
        end = self.y_test.shape[0]
        x_range = np.arange(0, end)

        if smooth:
            df_preds = pd.DataFrame(self.y_pred)
            df_trues = pd.DataFrame(self.y_test)
            self.y_pred = df_preds.rolling(200).mean().values
            self.y_test = df_trues.rolling(200).mean().values
        plt.plot(x_range, self.y_test, label='true')
        plt.plot(x_range, self.y_pred, label='pred')
        plt.legend()
        plt.show()

    def plot_predictions_train(self, smooth=False):
        end = self.y_train.shape[0]
        x_range = np.arange(0, end)

        if smooth:
            df_preds = pd.DataFrame(self.y_pred_train)
            df_trues = pd.DataFrame(self.y_train)
            self.y_pred_train = df_preds.rolling(200).mean().values
            self.y_train = df_trues.rolling(200).mean().values
        plt.plot(x_range, self.y_train, label='true')
        plt.plot(x_range, self.y_pred_train, label='pred')
        plt.legend()
        plt.show()

    def save_results(self, scores, filename):
        dest_path = Path(self.results_folder, filename)
        dest_path.with_suffix(".csv")

        if dest_path.exists():
            print("Modeling results for {} has already run".format(filename))
            return

        df = pd.DataFrame.from_dict(scores, orient='index')
        df.to_csv(dest_path.__str__())
        print("Saved results modeling for", filename)

    @staticmethod
    def _zero_prediction(x):
        if x <= 500.0:
            return 0
        else:
            return x


class MLEvaluator(BaseEvaluator):
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, regressor, zero_preds=True):
        super().__init__(x_train, x_val, x_test, y_train, y_val, y_test, zero_preds)
        self.regressor = regressor

    def inference(self, data):
        return self.regressor.predict(data)


class DLEvaluator(BaseEvaluator):
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, scaler, model, ckpt_path, zero_preds=True):
        super().__init__(x_train, x_val, x_test, y_train, y_val, y_test, scaler, zero_preds)
        self.model = model.load_from_checkpoint(ckpt_path)
        self.model.freeze()

    def inference(self, data):
        return self.model(torch.from_numpy(data).float())
