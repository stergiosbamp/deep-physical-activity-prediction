import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_HOURLY_DATASETS, BASE_PATH_DAILY_DATASETS

from src.model.dl.dataset import TimeSeriesDataset


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, seq_len=1, batch_size=128, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

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

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        if stage == 'fit' or stage is None:
            self.x_train = x_train
            self.y_train = y_train.values.reshape((-1, 1))
            # self.y_train = y_train.values

        if stage == 'test' or stage is None:
            self.x_test = x_test
            self.y_test = y_test.values.reshape((-1, 1))
            # self.y_test = y_test.values

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
