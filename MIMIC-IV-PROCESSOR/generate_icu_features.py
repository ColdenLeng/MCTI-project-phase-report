import os
import pandas as pd
import numpy as np

DATA_DIR = "."
SAVE_DIR = "./intermediate"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading icustays.csv.gz ...")
df = pd.read_csv(os.path.join(DATA_DIR, "icustays.csv.gz"),
                 usecols=["hadm_id", "intime", "outtime", "los"],
                 parse_dates=["intime", "outtime"])

df = df.dropna(subset=["hadm_id", "intime", "outtime", "los"])
df["hadm_id"] = df["hadm_id"].astype(int)

# ICU duration in hours
df["icu_duration_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0

agg = df.groupby("hadm_id").agg(
    icu_count = ("los", "count"),
    icu_total_los = ("los", "sum"),
    icu_avg_los = ("los", "mean"),
    icu_max_los = ("los", "max"),
    icu_min_los = ("los", "min"),
    icu_total_duration_hours = ("icu_duration_hours", "sum")
).reset_index()

out_path = os.path.join(SAVE_DIR, "icu_features.csv")
agg.to_csv(out_path, index=False)

print(f"[SUCCESS] Saved icu_features.csv to {out_path}, shape = {agg.shape}")
