import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report

from model import UltrasoundEncoder
from glob_model import GlobModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- CLASS NAMES --------
CLASS_NAMES = ["normal", "benign", "malignant"]
CLASS_TO_IDX = {
    "normal": 0,
    "benign": 1,
    "malignant": 2
}

# -------- IMAGE TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# -------- LOAD MODELS --------
encoder = UltrasoundEncoder().to(DEVICE)
encoder.load_state_dict(
    torch.load("ultrasound_encoder.pth", map_location=DEVICE),
    strict=False
)
encoder.eval()

global_model = GlobModel(input_dim=128, num_classes=3).to(DEVICE)
global_model.load_state_dict(
    torch.load("models/latest_global.pth", map_location=DEVICE)
)
global_model.eval()

# -------- DATA DIRECTORY --------
DATA_DIR = r"R:\final_proj\thyroid\ultrasound_organized"

total = 0
correct = 0
pred_count = {c: 0 for c in CLASS_NAMES}

y_true = []
y_pred = []

# -------- EVALUATION LOOP --------
with torch.no_grad():
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠️ Folder not found: {class_dir}")
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            image = Image.open(img_path).convert("L")
            image = transform(image).unsqueeze(0).to(DEVICE)

            features = encoder(image)
            logits = global_model(features)
            pred = torch.argmax(logits, dim=1).item()

            total += 1
            pred_count[CLASS_NAMES[pred]] += 1

            y_true.append(CLASS_TO_IDX[class_name])
            y_pred.append(pred)

            if pred == CLASS_TO_IDX[class_name]:
                correct += 1

# -------- BASIC RESULTS (UNCHANGED) --------
accuracy = (correct / total) * 100 if total > 0 else 0

print("\n================ BASIC EVALUATION =================")
print("Total images:", total)
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
print("Order: [normal, benign, malignant]\n")
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
