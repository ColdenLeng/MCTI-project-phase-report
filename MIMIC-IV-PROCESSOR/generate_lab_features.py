import os
import pandas as pd
import numpy as np

DATA_DIR = "."
SAVE_DIR = "./intermediate"
os.makedirs(SAVE_DIR, exist_ok=True)

TOP_N = 30

print("Loading labevents.csv.gz ...")
df = pd.read_csv(os.path.join(DATA_DIR, "labevents.csv.gz"),
                 usecols=["hadm_id", "itemid", "charttime", "valuenum", "flag"],
                 parse_dates=["charttime"])

df = df.dropna(subset=["valuenum"])
df["flag"] = df["flag"].fillna("")

top_items = df["itemid"].value_counts().nlargest(TOP_N).index.tolist()
print(f"Selected top {TOP_N} lab itemids: {top_items}")

feature_frames = []

for item in top_items:
    sub = df[df["itemid"] == item].copy()
    sub = sub.sort_values(by=["hadm_id", "charttime"])

    agg = sub.groupby("hadm_id").agg(
        lab_mean = ("valuenum", "mean"),
        lab_std = ("valuenum", "std"),
        lab_min = ("valuenum", "min"),
        lab_max = ("valuenum", "max"),
        lab_first = ("valuenum", lambda x: x.iloc[0]),
        lab_last = ("valuenum", lambda x: x.iloc[-1]),
        lab_count = ("valuenum", "count"),
        lab_abnormal = ("flag", lambda x: (x != "").sum())
    )

    agg["lab_trend"] = agg["lab_last"] - agg["lab_first"]

    agg.columns = [f"lab_{item}_{col}" for col in agg.columns]
    agg = agg.reset_index()

    feature_frames.append(agg)

from functools import reduce
df_final = reduce(lambda left, right: pd.merge(left, right, on="hadm_id", how="outer"), feature_frames)
df_final = df_final.fillna(0)

# 保存
out_path = os.path.join(SAVE_DIR, "lab_features.csv")
df_final.to_csv(out_path, index=False)

print(f"[SUCCESS] Saved lab_features.csv to {out_path}, shape = {df_final.shape}")
