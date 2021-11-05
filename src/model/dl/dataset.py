import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))

    def __len__(self):
        return self.X.__len__()

    def __getitem__(self, index):
        return self.X[index], self.y[index]
