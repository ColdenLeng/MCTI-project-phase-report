import os
import pandas as pd
import numpy as np

DATA_PATH = "./inputevents.csv.gz"  # or adjust to "./saved/inputevents.csv.gz"
SAVE_PATH = "./intermediate/input_features.csv"

print("Loading inputevents.csv.gz ...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# drop NAs in key columns
df = df.dropna(subset=["hadm_id", "amount", "starttime", "endtime"])

# Convert to datetime
df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")
df = df.dropna(subset=["starttime", "endtime"])

# Compute duration (in hours)
df["duration_hr"] = (df["endtime"] - df["starttime"]).dt.total_seconds() / 3600.0

# Group and aggregate per hadm_id
agg_df = df.groupby("hadm_id").agg(
    input_total_amount=("amount", "sum"),
    input_event_count=("amount", "count"),
    input_max_rate=("rate", "max"),
    input_avg_rate=("rate", "mean"),
    input_duration_hours=("duration_hr", "sum"),
    input_distinct_order_types=("ordercomponenttypedescription", pd.Series.nunique)
).reset_index()

# Fill NaNs with 0
agg_df = agg_df.fillna(0)

# Save
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
agg_df.to_csv(SAVE_PATH, index=False)
print(f"Saved input_features.csv with shape {agg_df.shape}")
