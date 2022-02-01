import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.model.dl.datamodule import TimeSeriesDataModule
from src.preprocessing.dataset import DatasetBuilder
from src.config.settings import GPU


class MLPRegressor(pl.LightningModule):
    def __init__(self, n_features, hidden_size, batch_size, dropout, learning_rate):
        super(MLPRegressor, self).__init__()
        self.save_hyperparameters()

        # Params
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=dropout)
        self.learning_rate = learning_rate

        # Architecture
        self.fc = nn.Linear(self.n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        # Metrics and logging
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.r2 = torchmetrics.regression.R2Score()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat, y)
        self.log("train_loss", {"MSE": mse, "MAE": mae, "R2": r2}, prog_bar=True, on_step=False, on_epoch=True)
        return mae

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat, y)
        self.log("val_loss", {"MSE": mse, "MAE": mae, "R2": r2}, prog_bar=True, on_step=False, on_epoch=True)
        return mae

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat, y)
        self.log("test_loss", {"MSE": mse, "MAE": mae, "R2": r2}, on_step=False, on_epoch=True)
        return mae


if __name__ == '__main__':
    pl.seed_everything(1)

    # Create dataset
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../../data/datasets/hourly/df-3x24-just-steps.pkl')

    dataset = ds_builder.create_dataset_steps_features()
    x_train, x_val, x_test, y_train, y_val, y_test = ds_builder.get_train_val_test(dataset, val_ratio=0.2)

    p = dict(
        batch_size=64,
        max_epochs=100,
        n_features=x_train.shape[1],
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
    model = MLPRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        batch_size=p['batch_size'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
    )

    model_checkpoint = ModelCheckpoint(
        filename='MLP-batch-{batch_size}-epoch-{max_epochs}-hidden-{hidden_size}-dropout-{'
                 'dropout}-lr-{learning_rate}'.format(**p)
    )

    # Trainer
    trainer = Trainer(max_epochs=p['max_epochs'], callbacks=[model_checkpoint], gpus=GPU)

    trainer.fit(model, dm)
