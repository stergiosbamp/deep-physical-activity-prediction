import pandas as pd


class Window:

    def __init__(self, n_in, n_out=1, dropna=True):
        """
        Constructor
        Args:
            n_in: The number of lagged observations.
            n_out: The number of output columns. Defaults to 1 for uni-variate time series problems.
            dropna: Whether to drop nan values due to non-existing previous data.
        """

        self.n_in = n_in
        self.n_out = n_out
        self.dropna = dropna

    def to_supervised_dataset(self, data: pd.DataFrame):
        """
        Function that turns a time series dataframe into a supervised
        dataset by using lagged observations.

        Args:
            data: The data in a timeseries format. E.g. consecutive values

        Returns:
            The supervised dataset. For uni-variate time series problems, the output column
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

    def aggregate_predictions(self, data, freq='1D'):
        """
        Function that aggregates predictions for the next day by default.
        This is useful in a way that we want to predict the next day's steps, rather than
        the steps of the next hour (if resampling by hour).

        It return only the first from the n_in records, because the others are already included
        in the aggregated value. So sliding in the next day's values is cheating.

        Essentially it provides tumbling windows.

        Args:
            data: The data to aggregate
            freq: The frequency of aggregation for resampling. Defaults to aggregation of next day's steps

        Returns:
            The dataset with output ('var1(t)' column) the aggregated steps count.
        """

        # find offset
        offset = data.index[0].hour
        offset_str = str(offset) + "H"

        # resample by offset, so days doesn't necessarily start from 00:00 (midnight)
        agg_stepscount = data['var1(t)'.format(self.n_in)].resample(rule=freq, offset=offset_str).sum()

        # drop hourly predictions
        data.drop(columns=['var1(t)'], inplace=True)

        # create again the prediction column to be populated
        # with the aggregated values of the next day
        data['var1(t)'] = None

        # populate it with aggregated predictions
        for i in range(0, agg_stepscount.shape[0]):
            start_date = agg_stepscount.index[i-1]
            end_date = agg_stepscount.index[i]

            data.at[start_date:end_date, 'var1(t)'] = agg_stepscount[start_date]

        data.at[end_date:, 'var1(t)'] = agg_stepscount[end_date]

        return data[::self.n_in]
