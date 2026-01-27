import torch
import numpy as np
import joblib

from model import BloodEncoder
from glob_model import GlobModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LABEL NAMES --------
CLASS_NAMES = {
    0: "normal",
    1: "hypo",
    2: "hyper"
}

# -------- LOAD SCALER --------
scaler = joblib.load("blood_scaler.pkl")

# -------- LOAD BLOOD ENCODER --------
encoder = BloodEncoder().to(DEVICE)
encoder.load_state_dict(
    torch.load("blood_encoder.pth", map_location=DEVICE)
)
encoder.eval()

# -------- LOAD GLOBAL MODEL --------
global_model = GlobModel().to(DEVICE)
global_model.load_state_dict(
    torch.load("models/latest_global.pth", map_location=DEVICE)
)
global_model.eval()

# -------- USER INPUT --------
print("Enter blood test values:")

TSH = float(input("TSH value: "))
T3 = float(input("T3 value: "))
T4 = float(input("T4 value: "))
Age = float(input("Age: "))
Gender = int(input("Gender (0 = Female, 1 = Male): "))

# -------- PREPARE INPUT --------
X = np.array([[TSH, T3, T4, Age, Gender]])
X = scaler.transform(X)   # IMPORTANT
X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

# -------- PREDICTION --------
with torch.no_grad():
    features = encoder(X)
    logits = global_model(features)
    pred_class = torch.argmax(logits, dim=1).item()

print("\nBlood Prediction:", CLASS_NAMES[pred_class])
