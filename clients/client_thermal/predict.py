import torch
from PIL import Image
import torchvision.transforms as transforms
import os

from model import ThermalEncoder
from glob_model import GlobModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Class labels (adjust order if needed) --------
CLASS_NAMES = {
    0: "Normal",
    1: "Benign",
    2: "Malignant"
}

# -------- Image preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------- Load thermal encoder --------
encoder = ThermalEncoder().to(DEVICE)
encoder.load_state_dict(
    torch.load("thermal_encoder.pth", map_location=DEVICE),
    strict=False
)
encoder.eval()

# -------- Load global federated model --------
global_model = GlobModel().to(DEVICE)
global_model.load_state_dict(
    torch.load("models/latest_global.pth", map_location=DEVICE)
)
global_model.eval()

# -------- Get image path from user --------
image_path = input("Enter thermal image path: ").strip()

if not os.path.exists(image_path):
    print("❌ Image path does not exist")
    exit()

# -------- Load image --------
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# -------- Prediction --------
with torch.no_grad():
    features = encoder(image)
    logits = global_model(features)
    pred_class = torch.argmax(logits, dim=1).item()

print("Prediction:", CLASS_NAMES[pred_class])
