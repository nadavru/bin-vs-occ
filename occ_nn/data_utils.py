import numpy as np
from torch.utils.data import Dataset

class PaperDataset(Dataset):
    def __init__(self, db, k):
        super().__init__()
        self.k = k
        self.db = db # np array
        self.total_rows = db.shape[0]
        self.d = db.shape[1]

    def __getitem__(self, ind):
        row = self.db[ind]
        a, b = [], []
        for i in range(self.d-self.k+1):
            a.append(row[i:i+self.k])
            b.append(np.concatenate((row[:i],row[i+self.k:])))
        a = np.stack(a)
        b = np.stack(b)
        
        return a, b

    def __len__(self):
        return self.total_rows
