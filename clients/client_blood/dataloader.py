import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class BloodDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # EXACT columns from cleaned dataset
        self.features = [
            "TSH",
            "T3",
            "T4",
            "Age",
            "Gender"
        ]

        X = self.df[self.features].values

        # Normalize features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
