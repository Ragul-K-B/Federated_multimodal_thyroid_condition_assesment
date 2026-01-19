import torch
from model import BloodEncoder
from dataloader import BloodDataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset (NO fitting scaler here)
dataset = BloodDataset(
    "R:\\final_proj\\thyroid\\clients\\client_blood\\blood_data.csv",
    train=False
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load trained encoder
encoder = BloodEncoder(input_dim=5, embedding_dim=128).to(DEVICE)
encoder.load_state_dict(torch.load("blood_encoder.pth", map_location=DEVICE))
encoder.eval()

features = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        emb = encoder(batch)
        features.append(emb.cpu())

features = torch.cat(features, dim=0)

torch.save(features, "blood_features.pt")
print("✅ Blood features extracted:", features.shape)
