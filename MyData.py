import torch
from torch.utils.data import DataLoader
import pandas as pd


class PrepareDataset(torch.utils.data.Dataset):

    def __init__(self, file_name, window):
        prepare_df = pd.read_csv(file_name)
        
        self.window = window

        x = prepare_df.iloc[:, 0:6].values
        y = prepare_df.iloc[:, 3:6].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train) - self.window

    def __getitem__(self, idx):
        x = self.x_train[idx:idx+self.window]
        return x, self.y_train[idx]


