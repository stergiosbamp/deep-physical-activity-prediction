import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from tqdm import tqdm

from .window import Window
from .preprocess import Preprocessor
from ..data.database import HealthKitDatabase


class DatasetBuilder:
    def __init__(self, n_in, save_dataset=True, directory='../../data/df_dataset_all_users.pkl', total_users=None):
        self.n_in = n_in
        self.save_dataset = save_dataset
        self.directory = Path(directory)
        self.total_users = total_users

        self.users_included = 0
        self.users_discarded = 0
        self.window = Window(n_in=self.n_in)
        self.hk_database = HealthKitDatabase()

    def get_train_test(self, ratio=0.75):
        if self.save_dataset and self.directory.exists():
            dataset = pd.read_pickle(self.directory.__str__())
        else:
            users = self.hk_database.get_all_healthkit_users()

            dataset = pd.DataFrame()

            for user in tqdm(users):
                cursor_results = self.hk_database.get_records_by_user(user_code=user)

                user_data = list(cursor_results)
                df_user = pd.DataFrame(user_data)

                preprocessor = Preprocessor(df=df_user)
                preprocessor\
                    .remove_outlier_dates() \
                    .resample_dates(frequency='1H')

                if not preprocessor.has_hourly_records(days_in_hours=self.n_in):
                    print("User {} has not aggregated hourly records for {} hours.".format(user, self.n_in))
                    self.users_discarded += 1
                    continue  # go to the next user and ignore the current one

                # current user has at least the requested amount of days
                self.users_included += 1

                # if requested a number of users
                if self.total_users is not None:
                    if self.users_included >= self.total_users:
                        break  # stop gathering records

                preprocessor\
                    .add_date_features()\
                    .add_sin_cos_features()

                df = preprocessor.df
                df = self.window.to_supervised_dataset(df)
                df = self.window.aggregate_predictions(df)
                # Append for each user to construct the final dataset
                dataset = dataset.append(df)

            # Re-sort based on the dates due to the mix of the different subjects
            dataset.sort_index(inplace=True)

            if self.save_dataset:
                # Save dataset into 'data' directory to run ML experiments without
                # requiring the whole preprocessing
                dataset.to_pickle(self.directory.__str__())

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
