import os
import torch
import pandas as pd
from dataloader import get_dataloader
from model import UltrasoundEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "ultrasound_organized")
MODEL_PATH = os.path.join(BASE_DIR, "ultrasound_encoder.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

loader = get_dataloader(DATA_DIR, batch_size=1, shuffle=False)

model = UltrasoundEncoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

features, labels = [], []

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        emb = model(imgs, extract_features=True)

        features.append(emb.cpu().numpy()[0])
        labels.append(lbls.item())

df = pd.DataFrame(features)
df["label"] = labels

OUT_CSV = os.path.join(BASE_DIR, "ultrasound_features_with_labels.csv")
df.to_csv(OUT_CSV, index=False)

print("✅ Ultrasound feature extraction completed")
