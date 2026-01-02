import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CONFIGURATION
# =========================
BASE_PATH = r"R:\project\thermal"
IMAGE_SIZE = 64
IMAGES_PER_CLASS = 300   # change if needed

CLASSES = {
    "normal": {
        "hotspot": False,
        "base_temp": (33.5, 34.5),
        "strength": (0.0, 0.0)
    },
    "benign": {
        "hotspot": True,
        "base_temp": (34.0, 35.0),
        "strength": (0.5, 1.0)
    },
    "malignant": {
        "hotspot": True,
        "base_temp": (35.0, 36.0),
        "strength": (1.5, 2.5)
    }
}

# =========================
# THERMAL IMAGE GENERATOR
# =========================
def generate_thermal_image(
    size,
    base_temp_range,
    hotspot,
    hotspot_strength
):
    # Smooth baseline temperature
    base_temp = np.random.uniform(*base_temp_range)
    image = np.random.normal(base_temp, 0.1, (size, size))

    # Localized heat source (thyroid nodule)
    if hotspot:
        cx, cy = np.random.randint(20, 44), np.random.randint(20, 44)
        strength = np.random.uniform(*hotspot_strength)

        for x in range(size):
            for y in range(size):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                image[x, y] += strength * np.exp(-dist / 10)

    # Clamp to physiological range
    image = np.clip(image, 33.0, 37.5)

    return image

# =========================
# DATASET CREATION
# =========================
def create_dataset():
    for label, params in CLASSES.items():
        class_path = os.path.join(BASE_PATH, label)
        os.makedirs(class_path, exist_ok=True)

        for i in range(IMAGES_PER_CLASS):
            img = generate_thermal_image(
                size=IMAGE_SIZE,
                base_temp_range=params["base_temp"],
                hotspot=params["hotspot"],
                hotspot_strength=params["strength"]
            )

            filename = f"{label}_{i:04d}.png"
            filepath = os.path.join(class_path, filename)

            plt.imsave(filepath, img, cmap="inferno")

        print(f"[OK] {IMAGES_PER_CLASS} images saved for class '{label}'")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    create_dataset()
    print("\nDataset generation completed successfully.")
