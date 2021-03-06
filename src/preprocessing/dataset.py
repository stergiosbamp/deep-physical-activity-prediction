import pandas as pd

from sklearn.model_selection import train_test_split
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

    def create_dataset_all_features(self):
        """
        Function that creates the dataset using sliding and/or tumbling window for aggregated predictions.
        It iterates over every users' data and applies the appropriate data cleaning operations.
        The final dataset contains both date and cyclic features.

        Returns:
            (pd.DataFrame): The dataset as a DataFrame.
        """

        if self.save_dataset and self.directory.exists():
            print("Loading dataset: {}".format(str(self.directory)))

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
                # if self.granularity == '1H':
                #     # In the case of hourly resampling the daily data sources
                #     # result in 23 of 24 records with zeros.
                #     preprocessor\
                #         .remove_daily_sources()

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

                # preprocessor \
                    # .impute_zeros() \
                    # .add_date_features() \
                    # .add_sin_cos_features(keep_only_sin_cos_transforms=False)

                df = preprocessor.df

                # The aggregation of predictions for the next day, should only be done
                # when we have hourly records. Otherwise (i.e. when having daily records)
                # the returned results are returned chunked due to the tumbling window and thus
                # we lose records that are associated with each other as a time-series.
                if self.granularity == '1H':
                    df = self.window.tumbling_window(df)
                else:
                    df = self.window.sliding_window(df)

                # Remove no wear days
                # df = preprocessor.remove_no_wear_days(df)

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

    def get_train_test(self, dataset, train_ratio=0.75):
        """
        Returns train and test data respecting the chronological order of the time series dataset.

        Args:
            dataset (pd.DataFrame): The dataset to split.
            train_ratio (float): The ratio for training/testing.

        Returns:
            (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame): The x_train, x_test, y_train,
                y_test sub-datasets.
        """

        if self.classification:
            y = dataset['class']
            X = dataset.drop(columns=['class'])
        else:
            y = dataset['var1(t)']
            X = dataset.drop(columns=['var1(t)'])

        # Split into train and test with respect to the chronological order i.e. no shuffle
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, shuffle=False)

        return x_train, x_test, y_train, y_test

    def get_train_val_test(self, dataset, val_ratio=0.2):
        """
        Returns train, validation and test data respecting the chronological order of the time series dataset.

        It splits the initial training set into a new training and validation set.

        Args:
            dataset (pd.DataFrame): The dataset to split.
            val_ratio (float): The ratio for the validation set.

        Returns:
            (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame): The
            x_train, x_val, x_test, y_train, y_val, y_test sub-datasets.
        """

        x_train_val, x_test, y_train_val, y_test = self.get_train_test(dataset)

        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_ratio, shuffle=False)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def create_dataset_steps_features(self):
        """
        Function that filters from the original dataset that contains both steps + date + cyclic features,
        only the steps features.
        Filters only the "var1" features which refer to date features.

        Returns:
            (pd.DataFrame): The supervised dataset with the steps features only.
        """

        dataset = self.create_dataset_all_features()
        dataset = dataset.filter(regex='var1\(t.+')
        return dataset

    def create_dataset_steps_cyclic_features(self):
        """
        Function that filters from the original dataset that contains both steps + date + cyclic features,
        only the steps and cyclic features.
        Removes the date features which are "var2" to "var7".

        Returns:
            (pd.DataFrame): The supervised dataset with the steps and cyclic features only.
        """

        dataset = self.create_dataset_all_features()
        to_remove = []
        for t in range(self.n_in, 0, -1):
            for var in range(2, 8):
                feature = "var{}(t-{})".format(var, t)
                to_remove.append(feature)

        # and current features
        for var in range(2, 8):
            feature = "var{}(t)".format(var)
            to_remove.append(feature)

        dataset = dataset.drop(columns=to_remove, axis=1)
        return dataset

    def create_dataset_steps_date_features(self):
        """
        Function that filters from the original dataset that contains both steps + date + cyclic features,
        only the steps and date features.
        Removes the cyclic features which are "var8" to "var17".

        Returns:
            (pd.DataFrame): The supervised dataset with the steps and date features only.
        """

        dataset = self.create_dataset_all_features()
        to_remove = []
        for t in range(self.n_in, 0, -1):
            for var in range(8, 18):
                feature = "var{}(t-{})".format(var, t)
                to_remove.append(feature)

        # and current features
        for var in range(8, 18):
            feature = "var{}(t)".format(var)
            to_remove.append(feature)

        dataset = dataset.drop(columns=to_remove, axis=1)
        return dataset

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
