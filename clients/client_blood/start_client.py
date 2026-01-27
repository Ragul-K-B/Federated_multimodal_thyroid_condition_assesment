import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import flwr as fl
import torch

from client import BloodFLClient
from glob_model import GlobModel
from model import BloodEncoder          # ✅ FIX IS HERE
from dataloader import BloodDataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load local encoder --------
encoder = BloodEncoder()
encoder.load_state_dict(
    torch.load("blood_encoder.pth", map_location=DEVICE)
)
encoder.eval()

# -------- Load data --------
dataset = BloodDataset("blood_data.csv")
loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

x, y = next(iter(loader))
x = x.to(DEVICE)
y = y.to(DEVICE)

# -------- Feature extraction --------
with torch.no_grad():
    features = encoder(x)

# -------- Global federated model --------
global_model = GlobModel()

# -------- Flower client --------
client = BloodFLClient(global_model, features, y)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client
)
