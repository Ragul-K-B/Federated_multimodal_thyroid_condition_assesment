import os
import torch
from dataloader import get_dataloader
from model import UltrasoundEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "ultrasound_organized")

# ================= LOAD DATA =================
loader = get_dataloader(DATA_DIR)

# ================= LOAD ENCODER =================
encoder = UltrasoundEncoder().to(DEVICE)
encoder.load_state_dict(
    torch.load("ultrasound_encoder.pth", map_location=DEVICE)
)
encoder.eval()

all_features = []
all_labels = []

# ================= FEATURE EXTRACTION =================
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        features = encoder(imgs)   # (batch, 128)

        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

# ================= CONCAT =================
all_features = torch.cat(all_features, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print("Final feature shape:", all_features.shape)
print("Final labels shape:", all_labels.shape)

# ================= SAVE =================
torch.save(all_features, "ultrasound_features.pt")
torch.save(all_labels, "ultrasound_labels.pt")

print("✅ Features & labels saved successfully")
