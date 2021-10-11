import pandas as pd


class SlidingWindow:

    def __init__(self, n_in, n_out=1, dropna=True):
        self.n_in = n_in
        self.n_out = n_out
        self.dropna = dropna

    def to_supervised_dataset(self, data: pd.DataFrame):
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
