from sklearn.model_selection import train_test_split


class Dataset:

    """
    Class that provides sub-datasets for training ML models. It takes as input the dataset created from
    the UBIWEAR's pre-processor.

    Attributes:
        dataset (pd.DataFrame): The dataset created from Processor
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def get_train_test(self, train_ratio=0.75):
        """
        Returns train and test data respecting the chronological order of the time series dataset.

        Args:
            train_ratio (float): The ratio for training/testing.

        Returns:
            (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame): The x_train, x_test, y_train,
                y_test sub-datasets.
        """

        y = self.dataset['var1(t)']
        X = self.dataset.drop(columns=['var1(t)'])

        # Split into train and test with respect to the chronological order i.e. no shuffle
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, shuffle=False)

        return x_train, x_test, y_train, y_test

    def get_train_val_test(self, train_ratio=0.75, val_ratio=0.2):
        """
        Returns train, validation and test data respecting the chronological order of the time series dataset.

        It splits the initial training set into a new training and validation set.

        Args:
            train_ratio (float): The ratio for training/testing.
            val_ratio (float): The ratio for the validation set.

        Returns:
            (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame), (pd.DataFrame): The
            x_train, x_val, x_test, y_train, y_val, y_test sub-datasets.
        """

        x_train_val, x_test, y_train_val, y_test = self.get_train_test(train_ratio=train_ratio)

        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_ratio, shuffle=False)

        return x_train, x_val, x_test, y_train, y_val, y_test
