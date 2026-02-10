import pandas as pd
import numpy as np

file_path = "aligned_raw_data.xlsx"
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
mask_2020_onwards = df['Date'].dt.year >= 2020

col = 'Total_New_Deaths_file_6'

# 强制列为 float
df[col] = pd.to_numeric(df[col], errors='coerce')

# 将 0 当作缺失
df.loc[mask_2020_onwards, col] = df.loc[mask_2020_onwards, col].replace(0, np.nan)

# 插值
df.loc[mask_2020_onwards, col] = df.loc[mask_2020_onwards, col].interpolate(
    method='linear', limit_direction='both'
)

# 前向和后向填充
df.loc[mask_2020_onwards, col] = df.loc[mask_2020_onwards, col].fillna(method='ffill').fillna(method='bfill')

# 添加 ±3% 噪声
noise_ratio = 0.03
np.random.seed(42)
filled_values = df.loc[mask_2020_onwards, col].to_numpy(dtype=float)  # 转 numpy float
noise = np.random.uniform(-noise_ratio, noise_ratio, size=filled_values.shape)
df.loc[mask_2020_onwards, col] = filled_values * (1 + noise)

# 保存
df.to_excel("/Users/lucaslu/PycharmProjects/HiCIM/filled_Total_New_Deaths_2020_onwards_with_noise.xlsx", index=False)
print("补全完成。")
