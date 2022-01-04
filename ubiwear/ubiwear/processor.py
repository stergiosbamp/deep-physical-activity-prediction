import pandas as pd
import numpy as np

from pandas.api.types import is_datetime64_dtype, is_float_dtype, is_int64_dtype


class Processor:
    """
    Main class the pre-process time-series aggregated wearable data. The available methods of the class should be
    used in a chaining style. It also offers a "magic" method "process" that processes the data in a pre-defined
    suggested pipeline oriented for physical activity data.

    Attributes:
        df (pd.DataFrame): The dataframe holding the uni-variate time-series data.
    """

    def __init__(self, df):
        self.__check_correctness(df)
        self.df = df

    def process(self, granularity, q, impute_start=None, impute_end=None):
        """
        A "magic" all-in-one method that holds a proposed logic for a series of suggested pre-processing steps for
        aggregated wearable data.

        Args:
            granularity (str): The string in panda's offset style for resampling and aggregating the time series data.
            q (float): Value between 0 <= q <= 1, the quantile(s) to compute for outliers removal.
            impute_start (int): The starting hour in 24-hours format, for imputation of zeros.
            impute_end (int): The ending hour in 24-hours format, for imputation of zeros.

        Returns:
            (pd.DataFrame): The pre-processed dataset.
        """

        self\
            .remove_duplicate_values_at_same_timestamp()\
            .remove_nan() \
            .remove_outlier_values(q=q) \
            .resample_dates(frequency=granularity)

        # Imputation is only meaningful in hourly granularity
        if granularity == '1H':
            self.impute_zeros(start_hour=impute_start, end_hour=impute_end)

        self.add_date_features() \
            .add_sin_cos_features(keep_only_sin_cos_transforms=True) \

        processed_df = self.df
        return processed_df

    def remove_nan(self):
        """
        Removes NaT/NaN dates that break the resampling.

        Returns:
            (self).
        """

        # NaN dates are also represented as NaT (Not A Timestamp)
        self.df.dropna(inplace=True)
        return self

    def remove_outlier_values(self, q=0.05):
        """
        Removes outlier steps count data by using the quantile method.

        Args:
           q (float): Value between 0 <= q <= 1, the quantile(s) to compute

        Returns:
            (self).
        """

        q_low = self.df["value"].quantile(q)
        q_hi = self.df["value"].quantile(1 - q)
        self.df = self.df[(self.df["value"] < q_hi) & (self.df["value"] > q_low)]
        return self

    def resample_dates(self, frequency):
        """
        Resamples and aggregates the data in fixed intervals and sums the data.

        Args:
            frequency (str): The string in panda's offset style for resampling and aggregating the time series data

        Returns:
            (self).
        """
        # check if after outliers removal, df is empty then return the empty df
        # e.g. (the same object as is)
        if self.df.empty:
            return self

        self.df = self.df.resample(rule=frequency).sum()
        return self

    def impute_zeros(self, start_hour=8, end_hour=24):
        """
        Imputes zero steps values in specific times using the interpolation method.
        By default it imputes zeros between 08:00 to 24:00. If there are data between this time period, and
        zeros too, then interpolates those zeros.
        The intuition is not to impute every zero since the zero values in the midnight and first morning hours are
        realistic.
        This function has meaning and applies only when resampling by hour e.g. for hourly datasets and not for daily
        datasets.

        Args:
            start_hour (int): The starting hour in 24-hours format, for imputation of zeros.
            end_hour (int): The ending hour in 24-hours format, for imputation of zeros.

        Returns:
            (self).

        """

        mask = (self.df.index.hour >= start_hour) & (self.df.index.hour <= end_hour) & (self.df['value'] == 0)
        # Replace those zeros with NaN to be caught by the pandas' interpolate method.
        self.df.loc[mask] = np.nan
        self.df.interpolate(method='linear', inplace=True, limit_direction='forward')
        return self

    def remove_duplicate_values_at_same_timestamp(self):
        """
        Removes the exactly identical records based on the time (startTime) and the according value of steps.

        Returns:
            (self).
        """

        # Drop subsequent days that have
        # at the exact same timestamp with the exact same value of steps
        self.df.drop_duplicates(subset=['value'], keep='first', inplace=True)
        return self

    def add_date_features(self):
        """
        Adds date features to the user's data.

        Returns:
            (self).
        """

        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['week'] = self.df.index.isocalendar().week
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        self.df['day'] = self.df.index.day
        self.df['hour'] = self.df.index.hour

        return self

    def add_sin_cos_features(self, keep_only_sin_cos_transforms=True):
        """
        Adds sin/cos transformation as features from the plain date features, to encode the cyclical features.

        Args:
            keep_only_sin_cos_transforms (bool): Whether to keep only the sin/cos transformation features and drop
                the plain date features.

        Returns:
            (self).
        """

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

        if keep_only_sin_cos_transforms:
            self.df.drop(columns=[
                'dayofweek', 'week', 'month', 'year', 'day', 'hour'
            ], inplace=True)

        return self

    def has_enough_records(self, days_in_hours):
        """
        Method that finds if the user's data have the required number of days (in hours) (after resampling) to be used
        later in the window.

        Args:
            days_in_hours (int): The days in hours.

        Returns:
            (bool): Whether user has enough records to be used later in the window.
        """

        start = self.df.index[0]
        end = self.df.index[-1]

        diff = end - start
        required_days = days_in_hours / 24

        if diff.days >= required_days:
            return True
        return False

    @staticmethod
    def remove_no_wear_days(df):
        """
        Removes days that user didn't wear the tracking device.
        Based on literature, no wear days are defined as steps less than 500.

        Essentially this function keeps all user's steps that are above 500 steps.

        Returns:
            (pd.DataFrame): The DataFrame with removed the no wear days.
        """

        df = df[df['var1(t)'] >= 500.0]
        return df

    @staticmethod
    def _sin_transform(values):
        """
        Applies SIN transform to a series value.

        Args:
            values (pd.Series): A series to apply SIN transform on.
        Returns
            (pd.Series): The transformed series.
        """

        return np.sin(2 * np.pi * values / len(set(values)))

    @staticmethod
    def _cos_transform(values):
        """
        Applies COS transform to a series value.

        Args:
            values (pd.Series): A series to apply SIN transform on.
        Returns
            (pd.Series): The transformed series.
        """

        return np.cos(2 * np.pi * values / len(set(values)))

    @staticmethod
    def __check_correctness(df):
        """
        Private method that ensures that the passed dataframe has the proper formatting, in order to be processed
        by the framework.

        Args:
            df (pd.DataFrame): The dataframe to perform checks.

        """

        assert 'value' in df.columns, "Dataframe must contain a column 'value' representing the data of " \
                                      "the time-series."

        if not is_datetime64_dtype(df.index.dtype):
            raise Exception("The index of the passed dataframe must be datetime data type.")

        if not (is_float_dtype(df.dtypes['value']) or is_int64_dtype(df.dtypes['value'])):
            raise Exception("The data type of 'value' column must be float or int.")
