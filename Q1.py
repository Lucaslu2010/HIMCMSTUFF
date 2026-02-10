import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

# 1. 读取Excel文件
# 假设你的excel文件叫 unhappy_data.xlsx
file_path = ('normalized_result_copy.csv')          # ← 修改成你自己的文件名
df = pd.read_csv(file_path)

# 2. 检查数据结构
print("数据前几行：")
print(df.head())
print("\n列名：", df.columns.tolist())

# 假设结构是：
# 第一列：日期
# 第2~6列：五个不幸相关指标
date_col = df.columns[0]
indicator_cols = df.columns[1:6]         # 取第2到第6列（5个指标）

print("\n使用的指标列：", indicator_cols.tolist())

# 3. 提取数值部分
X = df[indicator_cols].values           # shape: (n_samples, 5)

# 处理缺失值（这里简单用均值填充，你也可以选择删除或插值）
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# 4. 计算均值向量和协方差矩阵（整个数据集的分布中心）
mean_vec = np.mean(X, axis=0)
cov_mat = np.cov(X, rowvar=False)

# 为了数值稳定性，加一点小正则项（可选，样本量很小时特别有用）
cov_mat += np.eye(cov_mat.shape[0]) * 1e-6

# 计算协方差矩阵的逆（Mahalanobis距离需要）
try:
    inv_cov = np.linalg.inv(cov_mat)
except np.linalg.LinAlgError:
    print("协方差矩阵奇异！尝试使用伪逆...")
    inv_cov = np.linalg.pinv(cov_mat)

# 5. 计算每个数据点到“平均状态”的马氏距离，作为不幸指数
# 马氏距离越大 → 越偏离“正常/平均”状态 → 不幸程度可能越高
mahal_distances = []

for i in range(len(X)):
    diff = X[i] - mean_vec
    dist = mahalanobis(diff, np.zeros(5), inv_cov)   # 或直接用 inv_cov @ diff
    mahal_distances.append(dist)

# 6. 放入dataframe
df['不幸指数（马氏距离）'] = mahal_distances

# 可选：做个简单标准化（0~100分），更直观
max_dist = max(mahal_distances)
if max_dist > 0:
    df['不幸指数（0-100）'] = (df['不幸指数（马氏距离）'] / max_dist) * 100
else:
    df['不幸指数（0-100）'] = 0

# 7. 显示结果
print("\n计算结果（前10行）：")
print(df[[date_col, *indicator_cols, '不幸指数（马氏距离）', '不幸指数（0-100）']].head(10))

# 8. 保存结果
output_path = 'unhappy_index_result.xlsx'
df.to_excel(output_path, index=False)
print(f"\n结果已保存至：{output_path}")