import pandas as pd

# -------- File paths --------
input_file = "blood_data.csv"
output_file = "blood_data_grouped.csv"

# -------- Target grouping (STRING LABELS) --------
label_map = {
    # Normal
    "S": "normal",
    "I": "normal",

    # Hypothyroid
    "G": "hypo",
    "R": "hypo",
    "B": "hypo",
    "N": "hypo",
    "D": "hypo",

    # Hyperthyroid
    "A": "hyper",
    "F": "hyper",
    "M": "hyper",
    "K": "hyper",
    "L": "hyper",
}

# -------- Load CSV --------
df = pd.read_csv(input_file)

# -------- Keep only valid targets --------
df = df[df["Target"].isin(label_map.keys())].copy()

# -------- Change Target values --------
df["Target"] = df["Target"].map(label_map)

# -------- Save new CSV --------
df.to_csv(output_file, index=False)

print("Saved grouped dataset as:", output_file)
print("\nTarget distribution:")
print(df["Target"].value_counts())
