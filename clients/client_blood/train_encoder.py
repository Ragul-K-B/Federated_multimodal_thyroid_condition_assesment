import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import BloodDataset
from model import BloodAutoEncoder

# ---------------- CONFIG ----------------
CSV_PATH = "R:\\final_proj\\thyroid\\clients\\client_blood\\blood_data.csv"
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

# Dataset & Loader
dataset = BloodDataset(CSV_PATH, train=True)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True   # 👈 IMPORTANT
)


# Model
model = BloodAutoEncoder(input_dim=5, embedding_dim=128).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()

    for batch in loader:
        batch = batch.to(DEVICE)

        recon, _ = model(batch)
        loss = criterion(recon, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Reconstruction Loss: {avg_loss:.6f}")

# Save ONLY encoder weights
torch.save(model.encoder.state_dict(), "blood_encoder.pth")
print("✅ Encoder saved as blood_encoder.pth")
