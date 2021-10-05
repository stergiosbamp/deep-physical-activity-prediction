import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from .window import SlidingWindow
from .preprocess import Preprocessor
from ..data.database import Database


class DatasetBuilder:
    def __init__(self, n_in):
        self.sliding_window = SlidingWindow(n_in=n_in)
        self.database = Database()

    def get_train_test(self, ratio=0.75):
        # Sample data for two users.
        # Sliding window per subject and then concat the results
        # TODO: Fetch users' records from the database

        df1 = pd.read_pickle('../../df_one_user.pkl')
        df2 = pd.read_pickle('../../user_with_2_faulty_records.pkl')
        dataset = pd.DataFrame()

        for df in [df1, df2]:
            preprocessor = Preprocessor(df=df)
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
