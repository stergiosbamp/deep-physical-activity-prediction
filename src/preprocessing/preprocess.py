import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, df):
        # Keep what we need and do some type casting
        self.df = df[['startTime', 'value']]
        self.df['value'] = df['value'].astype('float64')

    def resample_dates(self, frequency='1H'):
        self.df = self.df.resample(frequency, on='startTime').sum()
        return self

    def remove_outlier_dates(self):
        # Sort by the date to see the faulty dates
        self.df.sort_values('startTime', inplace=True)

        # Some missing dates were filled with 2021 year
        self.df = self.df[self.df['startTime'] < pd.to_datetime('2021')]

        # Some dates are with NaT values
        self.df.dropna(inplace=True)
        return self

    def remove_outlier_values(self, q=0.1):
        q_low = self.df["value"].quantile(q)
        q_hi = self.df["value"].quantile(1 - q)
        self.df = self.df[(self.df["value"] < q_hi) & (self.df["value"] > q_low)]
        return self

    def scale(self):
        pass

    def add_date_features(self):
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['week'] = self.df.index.week
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        self.df['day'] = self.df.index.day
        self.df['hour'] = self.df.index.hour

        return self

    def add_sin_cos_features(self):
        self.df['dayofweek_sin'] = self._sin_transform(self.df['dayofweek'])
        self.df['dayofweek_cos'] = self._cos_transform(self.df['dayofweek'])
        self.df['week_sin'] = self._sin_transform(self.df['week'])
        self.df['week_cos'] = self._cos_transform(self.df['week'])
        self.df['month_sin'] = self._sin_transform(self.df['month'])
        self.df['month_cos'] = self._cos_transform(self.df['month'])
        self.df['day_sin'] = self._sin_transform(self.df['day'])
        self.df['day_cos'] = self._cos_transform(self.df['day'])
        self.df['hour_sin'] = self._sin_transform(self.df['hour'])
        self.df['hour_cos'] = self._cos_transform(self.df['hour'])

        return self

    def filter_data_source(self):
        pass

    @staticmethod
    def _sin_transform(values):
        """
        Applies SIN transform to a series value
        :param values: A series to apply SIN transform on
        :return: The transformed series
        """
        return np.sin(2 * np.pi * values / len(set(values)))

    @staticmethod
    def _cos_transform(values):
        """
        Applies COS transform to a series value
        :param values: A series to apply COS transform on
        :return: The transformed series
        """
        return np.cos(2 * np.pi * values / len(set(values)))
