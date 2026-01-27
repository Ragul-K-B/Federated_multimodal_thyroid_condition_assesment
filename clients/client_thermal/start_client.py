import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import flwr as fl
import torch

from client import ThermalFLClient
from glob_model import GlobModel
from model import ThermalEncoder
from dataloader import get_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD LOCAL ENCODER ----------------
encoder = ThermalEncoder()
encoder.load_state_dict(
    torch.load("thermal_encoder.pth", map_location=DEVICE),
    strict=False
)
encoder.to(DEVICE)
encoder.eval()

# ---------------- LOAD DATA (SMALL BATCH) ----------------
data_dir = r"R:\final_proj\thyroid\thermal"   # ✅ change only if needed
loader = get_dataloader(
    data_dir,
    batch_size=8,          # 🔥 IMPORTANT: SMALL BATCH
    shuffle=False
)

all_features = []
all_labels = []

# ---------------- FEATURE EXTRACTION (BATCH-WISE) ----------------
with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        features = encoder(images)     # SAFE
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

features = torch.cat(all_features, dim=0)
labels = torch.cat(all_labels, dim=0)

print(f"Extracted thermal features shape: {features.shape}")

# ---------------- GLOBAL FEDERATED MODEL ----------------
global_model = GlobModel()

# ---------------- START FL CLIENT ----------------
client = ThermalFLClient(global_model, features, labels)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client
)
