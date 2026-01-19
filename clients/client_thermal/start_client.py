import flwr as fl
import torch

from model import ThermalEncoder, ThermalClientModel
from dataloader import get_dataloader   # ✅ FIX
from client import ThermalFLClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load encoder --------
encoder = ThermalEncoder()
state_dict = torch.load("thermal_encoder.pth", map_location=DEVICE)

# remove classifier weights
state_dict.pop("classifier.weight", None)
state_dict.pop("classifier.bias", None)

encoder.load_state_dict(state_dict, strict=False)

encoder.to(DEVICE)

# -------- DataLoader (IMAGE-BASED) --------
# ⚠️ Change this path if needed
data_dir = "R:\\final_proj\\thyroid\\ultrasound_organized"
loader = get_dataloader(data_dir, batch_size=16, shuffle=True)

# -------- Model --------
model = ThermalClientModel(encoder, num_classes=3)

# -------- Flower Client --------
client = ThermalFLClient(model, loader, DEVICE)

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client
)
