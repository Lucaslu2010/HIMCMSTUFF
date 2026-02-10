import pandas as pd
import numpy as np
from scipy.spatial import distance


def combine_storm_columns(file_path, output_path):
    # 1. 加载数据 (增加对 Excel 的支持)
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # 2. 确定物理上的第 3 列和第 5 列
    # 注意：Python 索引 2 是第 3 列，索引 4 是第 5 列
    col3_name = df.columns[2]
    col5_name = df.columns[4]

    # 检查列名是否包含 "storm"
    if 'storm' not in col3_name.lower() or 'storm' not in col5_name.lower():
        print(f"跳过：列 '{col3_name}' 或 '{col5_name}' 不含 'storm'")
        return

    # 3. 数据清洗 (关键步骤：解决 SVD 不收敛问题)
    # 提取这两列数据
    subset = df[[col3_name, col5_name]].copy()

    # 将无法转换为数字的值设为 NaN，然后填充空值（这里使用均值填充，也可以用 0）
    subset = subset.apply(pd.to_numeric, errors='coerce')
    if subset.isnull().values.any():
        print("检测到空值或非数字字符，正在进行均值填充...")
        subset = subset.fillna(subset.mean())

    data_to_combine = subset.values

    # 4. 计算马氏距离
    try:
        mean_vec = np.mean(data_to_combine, axis=0)
        cov_matrix = np.cov(data_to_combine, rowvar=False)

        # 再次检查协方差矩阵是否包含 NaN 或 Inf
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            raise ValueError("协方差矩阵包含无效数值，请检查原始数据。")

        inv_cov_matrix = np.linalg.pinv(cov_matrix)

        mahal_dist = []
        for row in data_to_combine:
            d = distance.mahalanobis(row, mean_vec, inv_cov_matrix)
            mahal_dist.append(d)

        # 5. 更新 DataFrame
        # 将结果写入第三列
        df[col3_name] = mahal_dist
        # 删除第五列
        df.drop(columns=[col5_name], inplace=True)

        # 6. 保存结果
        df.to_csv(output_path, index=False)
        print(f"处理成功！结果已保存至 {output_path}")

    except Exception as e:
        print(f"计算失败: {e}")


if __name__ == "__main__":
    # 请确保已安装 openpyxl: pip install openpyxl
    combine_storm_columns('xiuzhengriqi_fixed.xlsx', 'result.csv')