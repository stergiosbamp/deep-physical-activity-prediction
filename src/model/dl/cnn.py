import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.config.settings import GPU
from src.model.dl.datamodule import TimeSeriesDataModule
from src.preprocessing.dataset import DatasetBuilder


class CNNRegressor(pl.LightningModule):
    def __init__(self, n_features, out_channels, hidden_size, batch_size, dropout, learning_rate,
                 kernel, padding):
        super(CNNRegressor, self).__init__()
        self.save_hyperparameters()

        self.n_features = n_features
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv1d(in_channels=self.n_features,
                               out_channels=self.out_channels,
                               kernel_size=kernel,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=kernel,
                               padding=padding)
        self.max_pool = nn.MaxPool1d(kernel_size=kernel)
        self.fc = nn.Linear(self.out_channels, 1)
        self.relu = nn.ReLU()

        # Metrics and logging
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.r2 = torchmetrics.regression.R2Score()

    def forward(self, x):
        # 3D for conv
        x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        # restore shape for feed-forward layers
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat, y)
        self.log("train_loss", {"MSE": mse, "MAE": mae, "R2": r2}, on_step=False, on_epoch=True)
        return mae

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat, y)
        self.log("val_loss", {"MSE": mse, "MAE": mae, "R2": r2}, on_step=False, on_epoch=True)
        return mae

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat, y)
        self.log("test_loss", {"MSE": mse, "MAE": mae, "R2": r2}, on_step=False, on_epoch=True)
        return mse


if __name__ == '__main__':
    pl.seed_everything(1)

    # Create dataset
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='1H',
                                save_dataset=True,
                                directory='../../../data/datasets/hourly/df-3x24-just-steps.pkl')

    dataset = ds_builder.create_dataset_steps_features()
    x_train, x_val, x_test, y_train, y_val, y_test = ds_builder.get_train_val_test(dataset, val_ratio=0.2)

    p = dict(
        batch_size=128,
        max_epochs=100,
        n_features=x_train.shape[1],
        out_channels=64,
        kernel=3,
        padding=2,
        hidden_size=100,
        dropout=0.2,
        learning_rate=0.0001,
        num_workers=4
    )

    # Init Data Module
    dm = TimeSeriesDataModule(
        x_train=x_train,
        x_test=x_test,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        batch_size=p['batch_size'],
        num_workers=p['num_workers']
    )

    # Init PyTorch model
    model = CNNRegressor(
        n_features=p['n_features'],
        out_channels=p['out_channels'],
        hidden_size=p['hidden_size'],
        batch_size=p['batch_size'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate'],
        kernel=p['kernel'],
        padding=p['padding']
    )

    model_checkpoint = ModelCheckpoint(
        filename='CNN-batch-{batch_size}-epoch-{max_epochs}-hidden-{hidden_size}-dropout-{'
                 'dropout}-lr-{learning_rate}-channels-{out_channels}-kernel-{kernel}-pad-{padding}'.format(**p)
    )

    # Trainer
    trainer = Trainer(max_epochs=p['max_epochs'], callbacks=[model_checkpoint], gpus=GPU)

    trainer.fit(model, dm)
