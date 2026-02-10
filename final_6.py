import pandas as pd
import numpy as np

# 读取 Excel 文件
file_path = "filled_Total_New_Deaths_2020_onwards.xlsx"  # 改成你的文件名
df = pd.read_excel(file_path)

# 确保 Date 列为 datetime 类型
df['Date'] = pd.to_datetime(df['Date'])

# 筛选 2020 年及以后的数据
mask_2020_onwards = df['Date'].dt.year >= 2020

# 处理 Total_New_Deaths_file_6
col = 'Total_New_Deaths_file_6'
# 将 0 替换为 NaN（表示缺失）
df.loc[mask_2020_onwards, col] = df.loc[mask_2020_onwards, col].replace(0, pd.NA)

# 转为 numeric
df[col] = pd.to_numeric(df[col], errors='coerce')

# 线性插值
df.loc[mask_2020_onwards, col] = df.loc[mask_2020_onwards, col].interpolate(
    method='linear', limit_direction='both'
)

# 前向和后向填充（处理首尾缺失）
df.loc[mask_2020_onwards, col] = df.loc[mask_2020_onwards, col].fillna(method='ffill').fillna(method='bfill')

# 添加 ±3% 随机噪声
noise_ratio = 0.03
np.random.seed(42)
df.loc[mask_2020_onwards, col] = df.loc[mask_2020_onwards, col] * (
    1 + np.random.uniform(-noise_ratio, noise_ratio, size=mask_2020_onwards.sum())
)

# 保存结果
df.to_excel("filled_Total_New_Deaths_2020_onwards_with_noise.xlsx", index=False)
print("Total_New_Deaths_file_6 从2020年开始的缺失值已补全，并添加了噪声。")
