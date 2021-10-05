import pandas as pd
from sklearn.datasets import make_regression


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


if __name__ == '__main__':
    X, y = make_regression(n_samples=30, n_features=2, random_state=1)
    df = pd.DataFrame(X)
    print("Original timeseries data")
    print(df)
    sliding_window = SlidingWindow(n_in=2)
    dataset = sliding_window.to_supervised_dataset(df)
    print("Transformed dataset to supervised learning")
    print(dataset)
