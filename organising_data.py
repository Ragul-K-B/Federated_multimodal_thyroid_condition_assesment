import os
import xml.etree.ElementTree as ET
import shutil

RAW_DIR = "ultrasound_data"
OUT_DIR = "ultrasound_organized"

os.makedirs(OUT_DIR, exist_ok=True)

def tirads_to_label(t):
    if t is None:
        return None
    t = t.lower().strip()
    if t in ["1", "2", "3"]:
        return "benign"
    if t in ["4a", "4b", "4c", "5"]:
        return "malignant"
    return None

for file in os.listdir(RAW_DIR):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(RAW_DIR, file))
    root = tree.getroot()

    image_id = root.find(".//image")
    tirads = root.find(".//tirads")

    if image_id is None:
        continue

    img_name = f"{file.replace('.xml','')}_{image_id.text}.jpg"

    # NORMAL case
    if tirads is None or tirads.text is None or root.find(".//mark") is None:
        label = "normal"
    else:
        label = tirads_to_label(tirads.text)

    if label is None:
        continue

    label_dir = os.path.join(OUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    img_path = os.path.join(RAW_DIR, img_name)
    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(label_dir, img_name))

print("✅ Ultrasound data organized successfully")
