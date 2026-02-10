import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. 模拟/读取数据
# 假设你的列名分别是 'date' 和 'trauma_level'
try:
    df = pd.read_excel('google_trends_Trauma_2016-01-01_2025-12-31.csv') # 或者 pd.read_csv
except:
    # 如果没有文件，创建一个演示数据集
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    # 模拟一个带上升趋势的数据：趋势 + 随机波动
    trend = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 2, 100)
    trauma_level = trend + noise
    df = pd.DataFrame({'date': dates, 'trauma_level': trauma_level})

# 2. 预处理
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 3. 执行去趋势 (Linear Detrending)
# scipy.signal.detrend 会计算最佳拟合直线并将其减去
df['detrended_trauma'] = signal.detrend(df['trauma_level'])

# 4. 可视化对比
plt.figure(figsize=(12, 6))

# 图1：原始趋势
plt.subplot(2, 1, 1)
plt.plot(df['date'], df['trauma_level'], label='Original (With Trend)', color='red')
plt.title('Original Trauma Level with Trend')
plt.legend()

# 图2：去趋势后的结果
plt.subplot(2, 1, 2)
plt.plot(df['date'], df['detrended_trauma'], label='Detrended (Fluctuations)', color='blue')
plt.axhline(0, color='black', linestyle='--', alpha=0.5) # 零基准线
plt.title('Deseasonalized/Detrended Trauma Level')
plt.legend()

plt.tight_layout()
plt.show()

# 5. 保存结果
df.to_excel('detrended_trauma_results.xlsx', index=False)
print("去趋势化处理完成，结果已保存。")