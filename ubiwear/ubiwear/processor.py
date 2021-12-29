import pandas as pd
import numpy as np

from pandas.api.types import is_datetime64_dtype, is_float_dtype, is_int64_dtype


class UbiwearProcessor:

    def __init__(self, df):
        """

        Args:
            df (pd.DataFrame): The dataframe representing the uni-variate time-series data.
        """
        self.__check_correctness(df)
        self.df = df

    @staticmethod
    def __check_correctness(df):
        assert 'value' in df.columns, "Dataframe must contain a column 'value' representing the data of " \
                                      "the time-series."

        if not is_datetime64_dtype(df.index.dtype):
            raise Exception("The index of the passed dataframe must be datetime data type.")

        if not (is_float_dtype(df.dtypes['value']) or is_int64_dtype(df.dtypes['value'])):
            raise Exception("The data type of 'value' column must be float or int.")
