import pandas as pd
import os

# ============ 配置 ============
folder_path = "data_folder"
output_file = "merged_shifted.xlsx"
file_names = ["file1.xlsx", "file2.xlsx", "file3.xlsx", "file4.xlsx", "file5.xlsx", "file6.xlsx"]
# ============================

all_data = []

for idx, file_name in enumerate(file_names):
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_excel(file_path, header=None)  # 假设没有表头
    n_rows = len(df)
    n_cols = 1 + len(file_names)  # 第一列 + 每个文件的值一列

    # 创建空的 DataFrame
    shifted_df = pd.DataFrame([[None]*n_cols for _ in range(n_rows)])

    # 第一列保持原样
    shifted_df[0] = df[0]

    # 第二列放在对应的列 idx+1
    shifted_df[idx+1] = df[1]

    all_data.append(shifted_df)

# 首尾连接
merged_df = pd.concat(all_data, ignore_index=True)
merged_df.to_excel(output_file, index=False, header=False)
print(f"✅ 文件已保存: {output_file}")
