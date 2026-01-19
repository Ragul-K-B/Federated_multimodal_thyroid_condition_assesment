import os
import torch
import torch.nn as nn
from dataloader import get_dataloader
from model import UltrasoundEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "ultrasound_organized")

device = "cuda" if torch.cuda.is_available() else "cpu"

loader = get_dataloader(DATA_DIR)
model = UltrasoundEncoder().to(device)

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

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

MODEL_PATH = os.path.join(BASE_DIR, "ultrasound_encoder.pth")
torch.save(model.state_dict(), MODEL_PATH)

print("✅ Ultrasound encoder trained and saved")
