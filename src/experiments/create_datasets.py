import os

from src.preprocessing.dataset import DatasetBuilder
from src.config.directory import BASE_PATH_DAILY_DATASETS, BASE_PATH_HOURLY_DATASETS


def create_and_store_dataset(n_in, directory, granularity):
    """
    Utility module's function that creates a variety of datasets.

    Args:
        n_in: (int): The number of lagged observations for the window construction.
        directory (str): The directory to save the dataset.
        granularity (str): The string in panda's offset style for resampling and aggregating the time series data.
    """

    dataset_builder = DatasetBuilder(n_in=n_in, granularity=granularity, save_dataset=True, directory=directory,
                                     total_users=None)
    dataset_builder.create_dataset_all_features()
    print("Users discarded {} due to not enough {} records".format(dataset_builder.users_discarded, n_in))


if __name__ == '__main__':
    # Hourly granularity datasets with imputation and no outlier values
    create_and_store_dataset(n_in=1 * 24,
                             directory='../../data/datasets/no-offset/df-1*24-just-steps.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=2 * 24,
                             directory='../../data/datasets/no-offset/df-2*24-just-steps.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=3 * 24,
                             directory='../../data/datasets/no-offset/df-3*24-just-steps.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=4 * 24,
                             directory='../../data/datasets/no-offset/df-4*24-just-steps.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=5 * 24,
                             directory='../../data/datasets/no-offset/df-5*24-just-steps.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=6 * 24,
                             directory='../../data/datasets/no-offset/df-6*24-just-steps.pkl',
                             granularity='1H')

    # Daily granularity datasets with imputation and no outlier values.
    # Note that lag observations here are 1, 2, 3, etc. because we resample and aggregate
    # by day and not by hour as before.
    create_and_store_dataset(n_in=1,
                             directory='../../data/datasets/no-offset/df-1-day-just-steps.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=2,
                             directory='../../data/datasets/no-offset/df-2-day-just-steps.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=3,
                             directory='../../data/datasets/no-offset/df-3-day-just-steps.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=4,
                             directory='../../data/datasets/no-offset/df-4-day-just-steps.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=5,
                             directory='../../data/datasets/no-offset/df-5-day-just-steps.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=6,
                             directory='../../data/datasets/no-offset/df-6-day-just-steps.pkl',
                             granularity='1D')
