import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.window import Window
from src.preprocessing.preprocess import Preprocessor
from src.data.database import HealthKitDatabase


class DatasetBuilder:
    """
    Class the constructs a dataset from the HealthKit data, to be used by ML models.

    Attributes:
        n_in: (int): The number of lagged observations for the window construction.

        granularity (str): The string in panda's offset style for resampling and aggregating the time series data.

        save_dataset (bool): Whether to save the resulted dataset.

        directory (str): The directory to save the dataset. It creates a pathlib.Path instance from this string.

        total_users (Union[int, None]): The amount of users to include for the construction of the dataset. If its
            None then construct the dataset for all available users.

        days_in_hours (int): How many days to use as lag for the sliding window. It's expressed in hours.
            1 day = 24 hours.

        users_included (int): How many users are included in the constructed dataset.

        users_discarded (int): How many users are discarded in the constructed dataset due to not enough records for
            the window.

        window (window.Window): The window class that can create sliding and tumbling windows.

        hk_database (database.HealthKitDatabase): A HealthKitDatabase instance for performing common operation to the
            collection.
    """

    def __init__(self, n_in, granularity, save_dataset=True, directory='../../data/df_dataset_all_users.pkl',
                 total_users=None):
        self.n_in = n_in
        self.granularity = granularity
        self.save_dataset = save_dataset
        self.directory = Path(directory)
        self.total_users = total_users
        self.days_in_hours = self.n_in if self.n_in % 24 == 0 else self.n_in * 24

        self.users_included = 0
        self.users_discarded = 0
        self.window = Window(n_in=self.n_in)
        self.hk_database = HealthKitDatabase()

    def create_dataset(self):
        """
        Function that creates the dataset using sliding and/or tumbling window for aggregated predictions.
        It iterates over every users' data and applies the appropriate data cleaning operations.

        Returns:
            (pd.DataFrame): The dataset as a DataFrame.
        """

        if self.save_dataset and self.directory.exists():
            dataset = pd.read_pickle(self.directory.__str__())

            # If it's loading an existing dataset, return it without subject which can't be processed by ML models.
            if self.save_dataset:
                dataset.drop(columns=['subject'], inplace=True)
        else:
            users = self.hk_database.get_all_healthkit_users()

            dataset = pd.DataFrame()

            for user in tqdm(users):
                cursor_results = self.hk_database.get_records_by_user(user_code=user)

                user_data = list(cursor_results)
                df_user = pd.DataFrame(user_data)

                preprocessor = Preprocessor(df=df_user)
                preprocessor \
                    .remove_duplicate_values_at_same_timestamp() \
                    .remove_outlier_dates() \
                    .remove_outlier_values(q=0.05) \
                    .resample_dates(frequency=self.granularity)

                if preprocessor.df.empty or (not preprocessor.has_enough_records(days_in_hours=self.days_in_hours)):
                    self.users_discarded += 1
                    continue  # go to the next user and ignore the current one

                # current user has at least the requested amount of days
                self.users_included += 1

                # if requested a number of users
                if self.total_users is not None:
                    if self.users_included >= self.total_users:
                        break  # stop gathering records

                preprocessor \
                    .impute_zeros() \
                    .add_date_features() \
                    .add_sin_cos_features()

                df = preprocessor.df
                df = self.window.to_supervised_dataset(df)

                df = self.window.aggregate_predictions(df)

                # If the dataset is to be saved, inject user (subject) id to know which records are from
                # whom (by sorting), if needed.
                if self.save_dataset:
                    df['subject'] = df_user['healthCode'].get(0)

                # Append for each user to construct the final dataset
                dataset = dataset.append(df)

            # Re-sort based on the dates due to the mix of the different subjects
            dataset.sort_index(inplace=True)

            if self.save_dataset:
                # Save dataset into 'data' directory to run ML experiments without
                # requiring the whole preprocessing
                dataset.to_pickle(self.directory.__str__())
        return dataset

    def get_train_test(self, ratio=0.75):
        """
        Returns train and test data respecting the chronological order of the time series dataset.

        Args:
            ratio (float): The ratio for training/testing.

        Returns:
            (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame): The x_train, x_test, y_train,
                y_test sub-datasets.
        """

        dataset = self.create_dataset()

        y = dataset['var1(t)']
        X = dataset.drop(columns=['var1(t)'])

        # Split into train and test with respect to the chronological order
        total_examples = dataset.shape[0]
        split_point = int(total_examples * ratio)

        x_train = X[:split_point]
        x_test = X[split_point:]

        y_train = y[:split_point]
        y_test = y[split_point:]

        return x_train, x_test, y_train, y_test

    def time_series_cv(self):
        pass


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
    dataset_builder.create_dataset()
    print("Users discarded {} due to not enough {} records".format(dataset_builder.users_discarded, n_in))


if __name__ == '__main__':
    # Hourly granularity datasets
    create_and_store_dataset(n_in=1 * 24,
                             directory='../../data/df-1*24-imputed-no-outliers-all-features-all-users-with-subject'
                                       '-injected.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=2 * 24,
                             directory='../../data/df-2*24-imputed-no-outliers-all-features-all-users-with-subject'
                                       '-injected.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=3 * 24,
                             directory='../../data/df-3*24-imputed-no-outliers-all-features-all-users-with-subject'
                                       '-injected.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=4 * 24,
                             directory='../../data/df-4*24-imputed-no-outliers-all-features-all-users-with-subject'
                                       '-injected.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=5 * 24,
                             directory='../../data/df-5*24-imputed-no-outliers-all-features-all-users-with-subject'
                                       '-injected.pkl',
                             granularity='1H')

    # Hourly granularity datasets
    create_and_store_dataset(n_in=1*24,
                             directory='../../data/df-1*24-all-features-all-users-with-subject-injected.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=2*24,
                             directory='../../data/df-2*24-all-features-all-users-with-subject-injected.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=3*24,
                             directory='../../data/df-3*24-all-features-all-users-with-subject-injected.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=4*24,
                             directory='../../data/df-4*24-all-features-all-users-with-subject-injected.pkl',
                             granularity='1H')
    create_and_store_dataset(n_in=5*24,
                             directory='../../data/df-5*24-all-features-all-users-with-subject-injected.pkl',
                             granularity='1H')

    # Daily granularity datasets. Note that lag observations here are 1, 2, 3, etc. because we resample and aggregate
    # by day and not by hour as before.
    create_and_store_dataset(n_in=1,
                             directory='../../data/df-1-day-all-features-all-users-with-subject-injected.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=2,
                             directory='../../data/df-2-day-all-features-all-users-with-subject-injected.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=3,
                             directory='../../data/df-3-day-all-features-all-users-with-subject-injected.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=4,
                             directory='../../data/df-4-day-all-features-all-users-with-subject-injected.pkl',
                             granularity='1D')
    create_and_store_dataset(n_in=5,
                             directory='../../data/df-5-day-all-features-all-users-with-subject-injected.pkl',
                             granularity='1D')
