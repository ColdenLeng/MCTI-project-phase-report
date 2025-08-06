import pandas as pd
import os
from collections import Counter

TOP_K = 50
SAVE_PATH = "./intermediate"
os.makedirs(SAVE_PATH, exist_ok=True)

df = pd.read_csv("diagnoses_icd.csv.gz", compression="gzip", low_memory=False)

top_codes = df['icd_code'].value_counts().nlargest(TOP_K).index.tolist()
df_top = df[df['icd_code'].isin(top_codes)].copy()

code_to_label = {code: i for i, code in enumerate(top_codes)}
df_top['label'] = df_top['icd_code'].map(code_to_label)

df_label = df_top[['hadm_id', 'label']].drop_duplicates()

label_counts = df_label['label'].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df_label = df_label[df_label['label'].isin(valid_labels)]

hadm_counts = df_label['hadm_id'].value_counts()
valid_hadm_ids = hadm_counts[hadm_counts > 1].index
df_label = df_label[df_label['hadm_id'].isin(valid_hadm_ids)]

unique_labels = sorted(df_label['label'].unique())
label_remap = {old: new for new, old in enumerate(unique_labels)}
df_label['label'] = df_label['label'].map(label_remap)

df_label.to_csv(os.path.join(SAVE_PATH, "label_df.csv"), index=False)

print(f"Final class count: {df_label['label'].nunique()}")
print(f"Label distribution:\n{df_label['label'].value_counts().sort_index()}")
print(f"Total unique hadm_id count: {df_label['hadm_id'].nunique()}")




