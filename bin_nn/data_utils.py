import numpy as np
from torch.utils.data import Dataset

class BinaryDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X # np array
        self.Y = Y # np array
        self.total_rows = X.shape[0]

    def __getitem__(self, ind):
        
        return self.X[ind], self.Y[ind]

    def __len__(self):
        return self.total_rows
