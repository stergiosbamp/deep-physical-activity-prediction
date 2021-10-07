import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

from .window import SlidingWindow
from .preprocess import Preprocessor
from ..data.database import HealthKitDatabase


class DatasetBuilder:
    def __init__(self, n_in, save_dataset=True, directory='../../data/df_dataset_all_users.pkl'):
        self.sliding_window = SlidingWindow(n_in=n_in)
        self.hk_database = HealthKitDatabase()
        self.save_dataset = save_dataset
        self.directory = Path(directory)

    def get_train_test(self, ratio=0.75):

        if self.save_dataset and self.directory.exists():
            dataset = pd.read_pickle(self.directory.__str__())
        else:
            users = self.hk_database.get_all_healthkit_users()

            dataset = pd.DataFrame()
            for user in users:
                cursor_results = self.hk_database.get_records_by_user(user_code=user)

                user_data = list(cursor_results)
                df_user = pd.DataFrame(user_data)

                preprocessor = Preprocessor(df=df_user)
                preprocessor\
                    .remove_outlier_dates()\
                    .resample_dates()\
                    .add_date_features()\
                    # .add_sin_cos_features()

                processed_df = preprocessor.df

                data = self.sliding_window.to_supervised_dataset(processed_df)

                # Append for each user to construct the final dataset
                dataset = dataset.append(data)

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
