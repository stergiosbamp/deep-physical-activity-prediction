import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_HOURLY_DATASETS, BASE_PATH_DAILY_DATASETS

from src.model.dl.dataset import TimeSeriesDataset


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, x_train, x_test, y_train, y_test, batch_size, num_workers=0):
        super().__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        scaler = MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.y_train = self.y_train.values.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.y_test = self.y_test.values.reshape((-1, 1))

    def train_dataloader(self):
        train_dataset = TimeSeriesDataset(self.x_train,
                                          self.y_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def test_dataloader(self):
        test_dataset = TimeSeriesDataset(self.x_test,
                                         self.y_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader