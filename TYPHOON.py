import pandas as pd

# ================= 配置 =================
FILE_PATH = 'Thunderstorm Wind (Wind 80 kts. and stronger).xlsx'  # 确保文件名正确


# =======================================

def process_storm_data():
    # 1. 读取数据
    print(f"正在读取 {FILE_PATH}...")
    df = pd.read_excel(FILE_PATH)

    # 2. 核心修复：处理脏数据
    # errors='coerce' 会把那句 "and a pivot irrigation..." 变成空值 (NaT)
    df['BEGIN_DATE'] = pd.to_datetime(df['BEGIN_DATE'], errors='coerce')

    # 记录并删除无法解析日期的行
    initial_count = len(df)
    df = df.dropna(subset=['BEGIN_DATE'])
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"⚠️ 自动过滤了 {dropped_count} 行非日期格式的脏数据。")

    # 3. 统计月度频率 (Frequency)
    # 我们以月为单位，统计 EVENT_ID 的出现次数
    df.set_index('BEGIN_DATE', inplace=True)
    monthly_data = df.resample('M')['EVENT_ID'].count().to_frame(name='actual_count')
    monthly_data['month'] = monthly_data.index.month
    monthly_data['year'] = monthly_data.index.year

    # 4. 计算季节性基准 (Seasonal Baseline)
    # 计算历史上每个月的平均发生频率（例如：所有 10 月的平均次数）
    seasonal_base = monthly_data.groupby('month')['actual_count'].mean().rename('seasonal_avg')

    # 5. 合并并去季节化 (计算纯增量)
    result = monthly_data.join(seasonal_base, on='month')
.,esult['seasonal_avg']

    # 6. 考虑增量趋势 (Trend)
    # 计算累积均值作为长期的趋势参考
    result['long_term_trend'] = result['actual_count'].expanding().mean()

    # 保存结果
    output_file = 'storm_analysis_results.xlsx'
    result.to_excel(output_file)

    print("-" * 30)
    print(f"✅ 处理成功！")
    print(f"结果预览（前5月）:\n", result[['actual_count', 'seasonal_avg', 'deseasonalized_increment']].head())
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    process_storm_data()