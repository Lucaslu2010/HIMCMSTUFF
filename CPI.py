import pandas as pd
import os

# ================= 配置区域 =================
INPUT_FILE = 'CPIAUCSL.xlsx'  # 你的原始数据
OUTPUT_FILE = 'trend_adjusted.xlsx'  # 处理后的数据


# ===========================================

def adjust_trend_growth():
    # 1. 加载数据
    if not os.path.exists(INPUT_FILE):
        # 自动生成演示数据
        data = {
            'observation_date': pd.date_range(start='2010-01-01', periods=10, freq='YS'),
            'CPIAUCSL': [100, 110, 121, 133, 146, 161, 177, 194, 214, 235]  # 模拟10%复合增长
        }
        pd.DataFrame(data).to_excel(INPUT_FILE, index=False)
        print(f"已创建示例文件: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values(by='observation_date')

    # 2. 确定基准点 (Base Point)
    # 我们通常以“第一天”为基准 (1.0)，观察后续相对于起点的增量
    base_value = df.iloc[0]['CPIAUCSL']
    base_date = df.iloc[0]['observation_date'].strftime('%Y-%m-%d')
    print(f"ℹ️ 趋势基准点: {base_date} (数值: {base_value})")

    # 3. 计算“增量倍数” (Growth Multiplier)
    # 这表示当前数值是基准点的多少倍
    df['Growth_Multiplier'] = df['CPIAUCSL'] / base_value

    # 4. 计算“去趋势系数” (Detrending Factor)
    # 逻辑：如果你想消除这个趋势，你需要乘以这个系数
    # 现在的 2.0 倍趋势，需要乘以 0.5 才能回到基准水平
    df['Detrend_Factor'] = base_value / df['CPIAUCSL']

    # 5. 示例：假设你有一列受趋势影响的随机数据 'Raw_Data'
    # 我们这里模拟一列数据来演示去趋势效果
    import numpy as np
    df['Raw_Data'] = 100 * df['Growth_Multiplier'] + np.random.randint(-5, 5, size=len(df))

    # 消除趋势后的数值
    df['Detrended_Value'] = df['Raw_Data'] * df['Detrend_Factor']

    # 6. 整理并保存
    df = df.round(4)
    df.to_excel(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"✅ 趋势调整完成！")
    print(f"结果已存至: {OUTPUT_FILE}")
    print("\n字段说明:")
    print("- Growth_Multiplier: 相比基准日增长了多少倍")
    print("- Detrend_Factor: 消除该趋势所需的缩放系数 (1/倍数)")
    print("- Detrended_Value: 消除趋势后的平稳数值")


if __name__ == "__main__":
    adjust_trend_growth()