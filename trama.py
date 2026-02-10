import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 读取 Excel
df = pd.read_csv('google_trends_Trauma_2016-01-01_2025-12-31.csv')

# 确保日期列是 datetime 类型
df['date'] = pd.to_datetime(df['date'])

# 将日期转换为数值（方便做回归）
df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)

# 拟合线性回归
X = df['date_ordinal'].values.reshape(-1, 1)
y = df['Trauma'].values
model = LinearRegression()
model.fit(X, y)

# 预测趋势
trend = model.predict(X)

# 去趋势
df['Trauma_detrended'] = df['Trauma'] - trend

# 保存结果
df.to_excel('data_detrended.xlsx', index=False)

print(df.head())
