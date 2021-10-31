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

        classification (bool): Whether to construct the dataset for a 5-class classification problem based on:
            "Tudor-Locke, Catrine, et al. "How many steps/day are enough?
            For adults." International Journal of Behavioral Nutrition and Physical Activity 8.1 (2011): 1-17."

            By default the dataset is constructed for regression.

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
                 total_users=None, classification=False):
        self.n_in = n_in
        self.granularity = granularity
        self.save_dataset = save_dataset
        self.directory = Path(directory)
        self.total_users = total_users
        self.classification = classification

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
                if 'subject' in dataset.columns:
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
                    .add_sin_cos_features(keep_only_sin_cos_transforms=True)

                df = preprocessor.df
                df = self.window.to_supervised_dataset(df)

                # The aggregation of predictions for the next day, should only be done
                # when we have hourly records. Otherwise (i.e. when having daily records)
                # the returned results are returned chunked due to the tumbling window and thus
                # we lose records that are associated with each other as a time-series.
                if self.granularity == '1H':
                    df = self.window.aggregate_predictions(df)

                # If the dataset is to be saved, inject user (subject) id to know which records are from
                # whom (by sorting), if needed.
                if self.save_dataset:
                    df['subject'] = df_user['healthCode'].get(0)

                # Append for each user to construct the final dataset
                dataset = dataset.append(df)

            # Re-sort based on the dates due to the mix of the different subjects
            dataset.sort_index(inplace=True)

            # Regression or classification
            if self.classification:
                # Create the class
                dataset['class'] = dataset['var1(t)'].apply(lambda x: self._regression_to_clf(x))
                # Drop the continuous target
                dataset.drop(columns=['var1(t)'], inplace=True)

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
            clf (bool): Whether to return train/test splits for classification


        Returns:
            (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame): The x_train, x_test, y_train,
                y_test sub-datasets.
        """

        dataset = self.create_dataset()

        if self.classification:
            y = dataset['class']
            X = dataset.drop(columns=['class'])
        else:
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

    @staticmethod
    def _regression_to_clf(x):
        if x < 5000:
            return 1
        elif 5000 <= x < 7500:
            return 2
        elif 7500 <= x < 10000:
            return 3
        elif 10000 <= x < 12500:
            return 4
        else:
            return 5
