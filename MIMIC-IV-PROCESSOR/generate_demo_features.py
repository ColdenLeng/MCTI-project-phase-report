import os
import pandas as pd

DATA_DIR = "."
SAVE_DIR = "./intermediate"
os.makedirs(SAVE_DIR, exist_ok=True)

admissions = pd.read_csv(os.path.join(DATA_DIR, "admissions.csv.gz"), 
                         usecols=["hadm_id", "subject_id", "admission_type"])
patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv.gz"), 
                       usecols=["subject_id", "gender", "anchor_age"])

df = pd.merge(admissions, patients, on="subject_id", how="left")

df["gender"] = df["gender"].map({"M": 1, "F": 0})

df = pd.get_dummies(df, columns=["admission_type"], prefix="admtype", drop_first=False)

onehot_cols = [col for col in df.columns if col.startswith("admtype_")]
df[onehot_cols] = df[onehot_cols].astype(int)

df = df.drop(columns=["subject_id"])

df = df.fillna(0)

out_path = os.path.join(SAVE_DIR, "demo_features.csv")
df.to_csv(out_path, index=False)

print(f"[SUCCESS] Saved demo_features.csv to {out_path} with shape {df.shape}")

