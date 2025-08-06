import pandas as pd
import numpy as np
import os

DATA_PATH = "./procedureevents.csv.gz"
SAVE_PATH = "./intermediate/procedure_features.csv"

print("Loading procedureevents.csv.gz ...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df = df.dropna(subset=["hadm_id", "starttime"])

df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")

df["duration_hour"] = (df["endtime"] - df["starttime"]).dt.total_seconds() / 3600.0
df["duration_hour"] = df["duration_hour"].fillna(0)

agg_df = df.groupby("hadm_id").agg(
    proc_event_count=("itemid", "count"),
    proc_distinct_itemid=("itemid", pd.Series.nunique),
    proc_total_duration=("duration_hour", "sum")
).reset_index()

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
agg_df.to_csv(SAVE_PATH, index=False)
print(f"Saved procedure_features.csv with shape {agg_df.shape}")
