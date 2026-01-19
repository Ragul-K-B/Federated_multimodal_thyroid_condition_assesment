import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class BloodDataset(Dataset):
    def __init__(self, csv_path, scaler_path="blood_scaler.pkl", train=True):
        df = pd.read_csv(csv_path)

        # -------- LABEL GROUPING MAP --------
        label_map = {
            # Normal
            "S": 0,
            "I": 0,

            # Hypothyroid
            "G": 1,
            "R": 1,
            "B": 1,
            "N": 1,
            "D": 1,

            # Hyperthyroid
            "A": 2,
            "F": 2,
            "M": 2,
            "K": 2,
            "L": 2,
        }

        # -------- FILTER VALID ROWS --------
        df = df[df["Target"].isin(label_map.keys())].copy()

        # -------- MAP LABELS --------
        y = df["Target"].map(label_map)

        # -------- FEATURES --------
        features = ["TSH", "T3", "T4", "Age", "Gender"]
        X = df[features].values

        # -------- SCALE FEATURES --------
        if train:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            joblib.dump(scaler, scaler_path)
        else:
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)

        # -------- TORCH TENSORS --------
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
