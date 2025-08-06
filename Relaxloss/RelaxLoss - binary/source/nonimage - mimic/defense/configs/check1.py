import pandas as pd
from collections import Counter

# 读取 CSV 文件
df = pd.read_csv("records_w_diag_icd10.csv")

# 丢弃没有诊断信息的行
df = df.dropna(subset=["all_diag_all"])

# 拆分每个样本的 ICD 列表
df['all_diag_all'] = df['all_diag_all'].str.split(';')

# 展平所有 ICD 编码到一个列表
all_labels = [label.strip() for sublist in df['all_diag_all'] for label in sublist]

# 统计每个 ICD 编码出现的次数
label_counter = Counter(all_labels)

# 取出现频率前 50 的 ICD 编码
top_50 = label_counter.most_common(50)

# 输出结果
for icd, count in top_50:
    print(f"{icd}: {count}")