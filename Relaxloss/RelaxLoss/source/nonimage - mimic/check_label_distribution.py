import pandas as pd

label_path = "./"

df = pd.read_csv(label_path)

num_classes = df["label"].nunique()
print(f"Total number of unique classes: {num_classes}")

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())
