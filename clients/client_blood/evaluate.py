import torch
import pandas as pd
import numpy as np

from model import BloodEncoder
from glob_model import GlobModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LABEL MAPPING --------
CLASS_NAMES = ["normal", "hypo", "hyper"]
LABEL_TO_IDX = {
    "normal": 0,
    "hypo": 1,
    "hyper": 2
}

# -------- LOAD DATA --------
CSV_PATH = "blood_data_grouped.csv"
FEATURES = ["TSH", "T3", "T4", "Age", "Gender"]

df = pd.read_csv(CSV_PATH)

X = df[FEATURES].values
y = df["Target"].map(LABEL_TO_IDX).values

# -------- LOAD SCALER --------
scaler = joblib.load("blood_scaler.pkl")
X = scaler.transform(X)

X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.long).to(DEVICE)

# -------- LOAD MODELS --------
encoder = BloodEncoder().to(DEVICE)
encoder.load_state_dict(
    torch.load("blood_encoder.pth", map_location=DEVICE)
)
encoder.eval()

global_model = GlobModel().to(DEVICE)
global_model.load_state_dict(
    torch.load("models/latest_global.pth", map_location=DEVICE)
)
global_model.eval()

# -------- EVALUATION --------
correct = 0
total = 0
pred_count = {c: 0 for c in CLASS_NAMES}

y_true = []
y_pred = []

with torch.no_grad():
    features = encoder(X)
    logits = global_model(features)
    preds = torch.argmax(logits, dim=1)

    for i in range(len(y)):
        total += 1

        true_label = y[i].item()
        pred_label = preds[i].item()

        y_true.append(true_label)
        y_pred.append(pred_label)

        pred_count[CLASS_NAMES[pred_label]] += 1

        if pred_label == true_label:
            correct += 1

# -------- BASIC RESULTS (UNCHANGED) --------
accuracy = (correct / total) * 100 if total > 0 else 0

print("\n================ BASIC EVALUATION =================")
print("Total samples:", total)
print("Correct predictions:", correct)
print(f"Accuracy: {accuracy:.2f}%")

print("\nPrediction distribution:")
for cls, count in pred_count.items():
    print(f"{cls}: {count}")

# -------- ADVANCED METRICS (ADDED AT END) --------
print("\n================ DETAILED METRICS =================")

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix")
print("(Rows = Actual, Columns = Predicted)")
print("Order: [normal, hypo, hyper]\n")
print(cm)

print("\nClassification Report")
print("(Precision / Recall / F1-score)\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

print("================ END OF REPORT =================\n")
