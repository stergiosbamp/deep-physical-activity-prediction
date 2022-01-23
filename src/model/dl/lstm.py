import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.config.settings import GPU
from src.model.dl.datamodule import TimeSeriesDataModule
from src.preprocessing.dataset import DatasetBuilder


class LSTMRegressor(pl.LightningModule):
    def __init__(self, n_features, hidden_size, batch_size, num_layers, dropout, learning_rate):
        super(LSTMRegressor, self).__init__()
        self.save_hyperparameters()

        # Params
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Architecture
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

        # Metrics and logging
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.r2 = torchmetrics.regression.R2Score()

    def forward(self, x):
        # reshape to pass each element of sequence through lstm, and not all together
        # LSTM needs a 3D tensor
        x = x.view(len(x), 1, -1)

        out, _ = self.lstm(x)
        out = self.relu(out)
        out = self.fc(out)

        # reshape back to be compatible with the true values' shape
        out = out.reshape(len(x), -1)
        return out

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
        return mse


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
        hidden_size=200,
        num_layers=3,
        dropout=0.3,
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
    model = LSTMRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        batch_size=p['batch_size'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
    )

    model_checkpoint = ModelCheckpoint(
        filename='LSTM-batch-{batch_size}-epoch-{max_epochs}-hidden-{hidden_size}-layers-{num_layers}-dropout-{'
                 'dropout}-lr-{learning_rate}'.format(**p)
    )

    # Trainer
    trainer = Trainer(max_epochs=p['max_epochs'], callbacks=[model_checkpoint], gpus=GPU)

    trainer.fit(model, dm)
