import os
import torch
import torch.nn as nn
from dataloader import get_dataloader
from model import UltrasoundEncoder

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "ultrasound_organized")

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ================= DATALOADER =================
loader = get_dataloader(DATA_DIR)

# ================= MODEL =================
encoder = UltrasoundEncoder().to(device)

# 🔥 TEMPORARY CLASSIFIER (ONLY FOR TRAINING)
classifier = nn.Linear(128, 3).to(device)

# ================= CLASS COUNTS (NEW DATASET) =================
normal = 164
benign = 292
malignant = 280
total = normal + benign + malignant

class_weights = torch.tensor(
    [total / normal, total / benign, total / malignant],
    dtype=torch.float
).to(device)

print("Class weights:", class_weights)

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(classifier.parameters()),
    lr=1e-3
)

# ================= TRAINING =================
EPOCHS = 10

for epoch in range(EPOCHS):
    encoder.train()
    classifier.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        features = encoder(imgs)      # 🔥 FEATURE LEARNING
        logits = classifier(features) # TEMP HEAD
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f}")

# ================= SAVE ENCODER ONLY =================
MODEL_PATH = os.path.join(BASE_DIR, "ultrasound_encoder.pth")
torch.save(encoder.state_dict(), MODEL_PATH)

print("✅ Encoder trained and saved at:", MODEL_PATH)
