import pandas as pd


class Window:
    """
    Class the performs window operations.

    Attributes:
        n_in (int): The number of lagged observations for the window construction.
        n_out (int): The number of output columns. Defaults to 1 for uni-variate time series problems.
        dropna (bool): Whether to drop nan values due to non-existing previous data.
    """

    def __init__(self, n_in, n_out=1, dropna=True):
        self.n_in = n_in
        self.n_out = n_out
        self.dropna = dropna

    def sliding_window(self, data):
        """
        Function that turns a time series dataframe into a supervised dataset by using lagged observations.

        Args:
            data (pd.DataFrame): The data in a timeseries format. E.g. consecutive values.

        Returns:
            (pd.DataFrame): The supervised dataset. For uni-variate time series problems, the output column
                is named 'var1(t)'.
        """

        n_vars = 1 if type(data) is list else data.shape[1]

        new_cols = []
        new_names = []
        for i in range(self.n_in, 0, -1):
            lagged_obs = data.shift(i)
            new_cols.append(lagged_obs)

            for j in range(n_vars):
                name = "var{}(t-{})".format(j + 1, i)
                new_names.append(name)

        for i in range(0, self.n_out):
            new_cols.append(data.shift(-i))
            if i == 0:
                for j in range(n_vars):
                    name = ('var{}(t)'.format(j + 1))
                    new_names.append(name)
            else:
                for j in range(n_vars):
                    name = ('var{}(t+{})'.format(j + 1, i))
                    new_names.append(name)

        result = pd.DataFrame()

        for col in new_cols:
            result = pd.concat([result, col], axis=1)

        result.columns = new_names
        if self.dropna:
            result.dropna(inplace=True)

        return result

    def tumbling_window(self, data, freq='1D'):
        """
        Function that aggregates predictions for the next day by default.
        This is useful in a way that we want to predict the next day's steps, rather than
        the steps of the next hour (if resampling by hour).

        It return only the first from the n_in records, because the others are already included
        in the aggregated value. So sliding in the next day's values is cheating.

        Essentially it provides tumbling windows.

        Args:
            data (pd.DataFrame): The data in a timeseries format. E.g. consecutive values.
            freq (str): The frequency in panda's style of aggregation for resampling. Defaults to aggregation
                of next day's steps

        Returns:
            (pd.DataFrame): The dataset with output ('var1(t)' column) the aggregated steps count.
        """

        sliding_dataset = self.sliding_window(data=data)

        # find each days' total steps
        steps_per_day = sliding_dataset['var1(t)'].resample(freq).sum()

        # slide to replicate the tumbling window with no overlapping records
        tumbling_dataset = sliding_dataset[::24].copy()

        # drop hourly predictions
        tumbling_dataset.drop(columns=['var1(t)'], inplace=True)

        # create again the prediction column to be populated
        # with the aggregated values of the next day
        tumbling_dataset['var1(t)'] = None

        # populate target steps with the total aggregated steps of next day
        for date_index, steps in zip(tumbling_dataset.index, steps_per_day):
            tumbling_dataset.at[date_index, 'var1(t)'] = steps

        return tumbling_dataset
