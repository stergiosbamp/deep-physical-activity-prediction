import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.config.directory import BASE_PATH_HOURLY_DATASETS, BASE_PATH_DAILY_DATASETS
from src.model.dl.datamodule import TimeSeriesDataModule
from src.preprocessing.dataset import DatasetBuilder


class CNNRegressor(pl.LightningModule):
    def __init__(self, n_features, hidden_size, batch_size, num_layers, dropout, learning_rate, criterion):
        super(CNNRegressor, self).__init__()
        self.save_hyperparameters()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.conv = nn.Conv1d(in_channels=self.n_features, out_channels=64, kernel_size=1)
        self.max_pool = nn.MaxPool1d(1)

        self.fc = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv(x)
        x = self.max_pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_loss = self.criterion(y_hat, y)
        self.log("loss", l1_loss)
        return l1_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_loss = self.criterion(y_hat, y)
        self.log("loss", l1_loss)
        return l1_loss


if __name__ == '__main__':
    pl.seed_everything(1)

    # Create dataset
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../../data/datasets/hourly/df-3*24-imputed-no-outliers-all-features-all'
                                          '-users-with-subject-injected.pkl')

    dataset = ds_builder.create_dataset_steps_features()
    X_train, X_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    p = dict(
        batch_size=128,
        criterion=nn.L1Loss(),
        max_epochs=8,
        n_features=X_train.shape[1],
        hidden_size=100,
        num_layers=3,
        dropout=0.2,
        learning_rate=0.001,
        num_workers=4
    )

    # Init Data Module
    dm = TimeSeriesDataModule(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
        batch_size=p['batch_size'],
        num_workers=p['num_workers']
    )

    # Init PyTorch model
    model = CNNRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
    )

    model_checkpoint = ModelCheckpoint(
        filename='CNN'
    )

    # Trainer
    trainer = Trainer(max_epochs=p['max_epochs'], callbacks=[model_checkpoint])

    trainer.fit(model, dm)
    trainer.test(model, dm)
