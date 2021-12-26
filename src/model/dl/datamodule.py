import pytorch_lightning as pl

from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from src.model.dl.dataset import TimeSeriesDataset


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, batch_size, num_workers=0):
        super().__init__()
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    # def prepare_data(self):
        # scaler = MinMaxScaler()
        # self.x_train = scaler.fit_transform(self.x_train)
        # self.x_val = scaler.transform(self.x_val)
        # self.x_test = scaler.transform(self.x_test)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.y_train = self.y_train.reshape((-1, 1))

        if stage == 'validate' or stage is None:
            self.y_val = self.y_val.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.y_test = self.y_test.reshape((-1, 1))

    def train_dataloader(self):
        train_dataset = TimeSeriesDataset(self.x_train,
                                          self.y_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = TimeSeriesDataset(self.x_val,
                                        self.y_val)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeSeriesDataset(self.x_test,
                                         self.y_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader
