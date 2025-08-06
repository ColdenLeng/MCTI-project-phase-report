import os
import pandas as pd
import numpy as np

DATA_PATH = "./outputevents.csv.gz"
SAVE_PATH = "./intermediate/output_features.csv"

print("Loading outputevents.csv.gz ...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Replace 'amount' with 'value'
df = df.dropna(subset=["hadm_id", "value", "charttime"])

# convert charttime
df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
df = df.dropna(subset=["charttime"])

# compute min/max time
duration_df = df.groupby("hadm_id").agg(
    output_first_time=("charttime", "min"),
    output_last_time=("charttime", "max")
)
duration_df["output_duration_hours"] = (
    (duration_df["output_last_time"] - duration_df["output_first_time"]).dt.total_seconds() / 3600.0
)
duration_df = duration_df[["output_duration_hours"]].reset_index()

# aggregate main features
agg_df = df.groupby("hadm_id").agg(
    output_total_amount=("value", "sum"),
    output_event_count=("value", "count"),
    output_max_amount=("value", "max"),
    output_avg_amount=("value", "mean"),
    output_distinct_items=("itemid", pd.Series.nunique)
).reset_index()

# merge duration + other stats
merged = pd.merge(agg_df, duration_df, on="hadm_id", how="left").fillna(0)

# save
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
merged.to_csv(SAVE_PATH, index=False)
print(f"Saved output_features.csv with shape {merged.shape}")

