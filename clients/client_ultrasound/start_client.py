import flwr as fl
import torch

from model import UltrasoundEncoder, UltrasoundClientModel
from dataloader import get_dataloader
from client import UltrasoundFLClient


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load encoder (IGNORE CLASSIFIER WEIGHTS) --------
encoder = UltrasoundEncoder().to(DEVICE)

state_dict = torch.load("ultrasound_encoder.pth", map_location=DEVICE)

# 🔥 Remove classifier weights safely
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith("classifier.")
}

encoder.load_state_dict(filtered_state_dict, strict=False)

# -------- DataLoader --------
data_dir = r"R:\final_proj\thyroid\ultrasound_organized"
loader = get_dataloader(data_dir, batch_size=16, shuffle=True)

# -------- Model (Encoder + Federated Classifier) --------
model = UltrasoundClientModel(encoder, num_classes=3).to(DEVICE)

# -------- Flower Client --------
client = UltrasoundFLClient(model, loader, DEVICE)

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client
)
