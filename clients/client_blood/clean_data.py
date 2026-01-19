import pandas as pd

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("R:\\final_proj\\thyroid\\thyroidDF.csv")

# ===============================
# 2. Standardize column names
# ===============================
df.columns = df.columns.str.strip().str.lower()

# ===============================
# 3. Rename required columns
# ===============================
df = df.rename(columns={
    "age": "Age",
    "sex": "Gender",
    "tsh": "TSH",
    "t3": "T3",
    "tt4": "T4",
    "target": "Target"
})

# ===============================
# 4. Keep ONLY required columns
# ===============================
required_cols = ["Age", "Gender", "TSH", "T3", "T4", "Target"]
df = df[required_cols]

# ===============================
# 5. Encode Gender
# ===============================
df["Gender"] = df["Gender"].map({"F": 0, "M": 1})

# ===============================
# 6. Convert numeric columns safely
# ===============================
num_cols = ["Age", "TSH", "T3", "T4"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ===============================
# 7. Handle missing FEATURE values
#    (median is medically safe)
# ===============================
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Drop rows where Gender is still missing
df = df.dropna(subset=["Gender"])

# ===============================
# 8. DATASET 1 — FEATURES ONLY
# ===============================
features_only = df[["Age", "Gender", "TSH", "T3", "T4"]]
features_only.to_csv("blood_data.csv", index=False)

# ===============================
# 9. DATASET 2 — FEATURES + LABELS
#    Keep only valid targets
# ===============================
labeled_df = df[
    df["Target"].notna() &
    (df["Target"] != "-")
].copy()

labeled_df.to_csv("blood_data.csv", index=False)

# ===============================
# 10. Summary
# ===============================
print("✅ Cleaning & splitting completed\n")

print("📁 blood_features_only.csv")
print("   Shape:", features_only.shape)

print("\n📁 blood_data.csv")
print("   Shape:", labeled_df.shape)
print("\nLabel distribution:")
print(labeled_df["Target"].value_counts())
