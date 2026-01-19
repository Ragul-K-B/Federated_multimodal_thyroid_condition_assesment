import os
import torch
import torch.nn as nn
from dataloader import get_dataloader
from model import ThermalEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "thermal")

device = "cuda" if torch.cuda.is_available() else "cpu"

loader = get_dataloader(DATA_DIR)
model = ThermalEncoder().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), os.path.join(BASE_DIR, "thermal_encoder.pth"))
print("✅ Thermal encoder trained and saved")
