import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data, self.targets = self._read_dataset()
        
    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.targets[idx]
    
    def __len__(self):
        return len(self.targets)
    
    def _read_dataset(self):
        df = pd.read_csv(self.path, names=np.arange(0, 188, 1))
    
        data = df.drop(187, axis=1).to_numpy()
        targets = df[187]
        return torch.tensor(data, dtype=torch.float), torch.tensor(targets, dtype=int)
    