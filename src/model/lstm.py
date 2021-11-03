import numpy
import os
import pandas
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_HOURLY_DATASETS, BASE_PATH_DAILY_DATASETS


class TimeseriesDataset(Dataset):
    def __init__(self, X, y, seq_len: int = 1):
        self.X = torch.tensor(X.values.astype(numpy.float32))
        self.y = torch.tensor(y.astype(numpy.float32))
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return self.X[index:index + self.seq_len], self.y[index+self.seq_len-1]


class TimeseriesDataModule(pl.LightningDataModule):
    def __init__(self, seq_len=1, batch_size=128, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.columns = None
        self.preprocessing = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' and self.x_train is not None:
            return
        if stage == 'test' and self.x_test is not None:
            return
        if stage is None and self.x_train is not None and self.x_test is not None:
            return

        ds_builder = DatasetBuilder(n_in=3 * 24,
                                    granularity='whatever',
                                    save_dataset=True,
                                    directory=os.path.join(BASE_PATH_HOURLY_DATASETS,
                                                           'df-3*24-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'))

        dataset = ds_builder.create_dataset_steps_features()
        x_train, x_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

        if stage == 'fit' or stage is None:
            self.x_train = x_train
            self.y_train = y_train.values.reshape((-1, 1))
            # self.y_train = y_train.values

        if stage == 'test' or stage is None:
            self.x_test = x_test
            self.y_test = y_test.values.reshape((-1, 1))
            # self.y_test = y_test.values

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.x_train,
                                          self.y_train,
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    # def val_dataloader(self):
    #     val_dataset = TimeseriesDataset(self.X_val,
    #                                     self.y_val,
    #                                     seq_len=self.seq_len)
    #     val_loader = DataLoader(val_dataset,
    #                             batch_size=self.batch_size,
    #                             shuffle=False,
    #                             num_workers=self.num_workers)
    #
    #     return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.x_test,
                                         self.y_test,
                                         seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader


class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # result = pl.TrainResult(loss)
        self.log('train_loss', loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y)
    #     # result = pl.EvalResult(checkpoint_on=loss)
    #     self.log('val_loss', loss)
    #     return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # result = pl.EvalResult()
        self.log('test_loss', loss)
        return loss


if __name__ == '__main__':
    p = dict(
        seq_len=72,
        batch_size=70,
        criterion=nn.L1Loss(),
        max_epochs=10,
        n_features=72,
        hidden_size=100,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
    )

    seed_everything(1)

    csv_logger = CSVLogger('./', name='lstm', version='1'),

    trainer = Trainer(
        max_epochs=p['max_epochs'],
        fast_dev_run=True,
        logger=csv_logger,
        progress_bar_refresh_rate=2,
    )

    model = LSTMRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        seq_len=p['seq_len'],
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
    )

    dm = TimeseriesDataModule(
        seq_len=p['seq_len'],
        batch_size=p['batch_size']
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
