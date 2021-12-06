import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.config.settings import GPU
from src.model.dl.datamodule import TimeSeriesDataModule
from src.preprocessing.dataset import DatasetBuilder


class LSTMRegressor(pl.LightningModule):
    def __init__(self, n_features, hidden_size, batch_size, num_layers, dropout, learning_rate, criterion):
        super(LSTMRegressor, self).__init__()
        self.save_hyperparameters()

        self.n_features = n_features
        self.hidden_size = hidden_size
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
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # reshape to pass each element of sequence through lstm, and not all together
        # LSTM needs a 3D tensor
        x = x.view(len(x), 1, -1)

        out, _ = self.lstm(x)
        out = out.reshape(-1, self.hidden_size)
        out = self.relu(out)
        out = self.fc(out)

        # reshape back to be compatible with the true values' shape
        out = out.reshape(len(x), 1)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.reshape(-1)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("loss", loss)
        return loss


if __name__ == '__main__':
    pl.seed_everything(1)

    # Create dataset
    ds_builder = DatasetBuilder(n_in=3*24,
                                granularity='whatever',
                                save_dataset=True,
                                directory='../../../data/datasets/variations/df-3x24-no-wear-days-500-just-steps.pkl')

    dataset = ds_builder.create_dataset_steps_features()
    x_train, x_val, x_test, y_train, y_val, y_test = ds_builder.get_train_val_test(dataset, val_ratio=0.2)

    p = dict(
        batch_size=128,
        criterion=nn.MSELoss(),
        max_epochs=30,
        n_features=x_train.shape[1],
        hidden_size=100,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
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
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
    )

    model_checkpoint = ModelCheckpoint(
        filename='2-stack-LSTM'
    )

    # Trainer
    trainer = Trainer(max_epochs=p['max_epochs'], callbacks=[model_checkpoint], gpus=0)

    trainer.fit(model, dm)
