import torch
import flwr as fl
from model import UltrasoundEncoder
from client import UltrasoundFLClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== LOAD DATA =====
features = torch.load("ultrasound_features.pt")
labels = torch.load("ultrasound_labels.pt")

print("Extracted ultrasound features shape:", features.shape)

# ===== START CLIENT =====
client = UltrasoundFLClient(features, labels)

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client
)
