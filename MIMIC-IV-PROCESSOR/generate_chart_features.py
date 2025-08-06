import os
import pandas as pd
import numpy as np

DATA_DIR = "."
SAVE_DIR = "./intermediate"
os.makedirs(SAVE_DIR, exist_ok=True)

TOP_N = 30

print("Loading chartevents.csv.gz ...")
df = pd.read_csv(os.path.join(DATA_DIR, "chartevents.csv.gz"),
                 usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                 parse_dates=["charttime"])

df = df.dropna(subset=["valuenum"])
df = df[df["hadm_id"].notna()]
df["hadm_id"] = df["hadm_id"].astype(int)

top_items = df["itemid"].value_counts().nlargest(TOP_N).index.tolist()
print(f"Selected top {TOP_N} chart itemids: {top_items}")

feature_frames = []

for item in top_items:
    sub = df[df["itemid"] == item].copy()
    sub = sub.sort_values(by=["hadm_id", "charttime"])

    agg = sub.groupby("hadm_id").agg(
        chart_mean = ("valuenum", "mean"),
        chart_std = ("valuenum", "std"),
        chart_min = ("valuenum", "min"),
        chart_max = ("valuenum", "max"),
        chart_first = ("valuenum", lambda x: x.iloc[0]),
        chart_last = ("valuenum", lambda x: x.iloc[-1]),
        chart_count = ("valuenum", "count")
    )

    agg["chart_trend"] = agg["chart_last"] - agg["chart_first"]
    agg.columns = [f"chart_{item}_{col}" for col in agg.columns]
    agg = agg.reset_index()
    feature_frames.append(agg)

from functools import reduce
df_final = reduce(lambda left, right: pd.merge(left, right, on="hadm_id", how="outer"), feature_frames)
df_final = df_final.fillna(0)

out_path = os.path.join(SAVE_DIR, "chart_features.csv")
df_final.to_csv(out_path, index=False)

print(f"[SUCCESS] Saved chart_features.csv to {out_path}, shape = {df_final.shape}")
