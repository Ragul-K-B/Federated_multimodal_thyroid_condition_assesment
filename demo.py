import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, rotate

# =========================
# CONFIG
# =========================
BASE_PATH = r"R:\final_proj\thyroid\thermal"
IMAGE_SIZE = 64
IMAGES_PER_CLASS = 300
LABELS = ["normal", "benign", "malignant"]

# =========================
# UTILS
# =========================
def gaussian_2d(x, y, cx, cy, sx, sy):
    return np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) +
                     ((y - cy) ** 2) / (2 * sy ** 2)))

def neck_mask(size):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    mask = gaussian_2d(
        x, y,
        cx=size//2,
        cy=size//2 + 6,
        sx=18,
        sy=26
    )
    return mask

# =========================
# THERMAL GENERATOR
# =========================
def generate_realistic_thermal_image(size=64, label="normal"):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    center = size // 2

    # ---- Background (cool) ----
    image = np.random.normal(32.8, 0.15, (size, size))

    # ---- Neck region ----
    neck = neck_mask(size)
    image += neck * np.random.uniform(0.8, 1.2)

    # ---- Vertical thermal gradient ----
    gradient = (y / size) * np.random.uniform(0.4, 0.7)
    image += gradient

    # ---- Thyroid lobes ----
    left_lobe = gaussian_2d(x, y, center - 9, center, 6, 8)
    right_lobe = gaussian_2d(x, y, center + 9, center, 6, 8)

    image += 0.6 * (left_lobe + right_lobe)

    # ---- Pathology ----
    if label == "benign":
        image += 0.25 * (left_lobe + right_lobe)

    elif label == "malignant":
        image += 0.9 * right_lobe
        image += np.random.normal(0, 0.08, image.shape)

    # ---- Camera blur (VERY important) ----
    image = gaussian_filter(image, sigma=1.2)

    # ---- Small rotation (camera misalignment) ----
    angle = np.random.uniform(-3, 3)
    image = rotate(image, angle, reshape=False, mode="nearest")

    # ---- Clamp ----
    image = np.clip(image, 33.0, 37.5)

    return image

# =========================
# DATASET CREATION
# =========================
def create_dataset():
    for label in LABELS:
        path = os.path.join(BASE_PATH, label)
        os.makedirs(path, exist_ok=True)

        for i in range(IMAGES_PER_CLASS):
            img = generate_realistic_thermal_image(label=label)
            filename = f"{label}_{i:04d}.png"
            plt.imsave(os.path.join(path, filename), img, cmap="inferno")

        print(f"[OK] {label} completed")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    create_dataset()
    print("\n✅ Dataset generated with improved realism.")
