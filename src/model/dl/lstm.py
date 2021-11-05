import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.model.dl.datamodule import TimeSeriesDataModule
from src.config.directory import LOGS_DIR


class LSTMRegressor(pl.LightningModule):
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
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # reshape to pass each element of sequence through lstm, and not all together
        # LSTM needs a 3D tensor
        x = x.view(len(x), 1, -1)

        out, _ = self.lstm(x)
        out = self.fc(out)
        out = self.fc2(out)

        # reshape back to be compatible with the true values' shape
        out = out.reshape(len(x), 1)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss


if __name__ == '__main__':
    p = dict(
        seq_len=72,
        batch_size=128,
        criterion=nn.L1Loss(),
        max_epochs=10,
        n_features=72,
        hidden_size=100,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        num_workers=4
    )

    seed_everything(1)

    csv_logger = CSVLogger(save_dir=os.path.join(LOGS_DIR, 'LSTM')),

    trainer = Trainer(
        max_epochs=p['max_epochs'],
        fast_dev_run=False,
        logger=csv_logger,
        progress_bar_refresh_rate=2,
        callbacks=[EarlyStopping(monitor="train_loss")]
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

    dm = TimeSeriesDataModule(
        seq_len=p['seq_len'],
        batch_size=p['batch_size'],
        num_workers=p['num_workers']
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
