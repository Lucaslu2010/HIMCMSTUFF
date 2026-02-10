import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据接入部分
# ==========================================
# 请将 'your_data.csv' 替换为您导出的文件名
# 确保数据中的 Date 列是日期格式
try:
    df = pd.read_csv('normalized_result.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    print("数据加载成功，样本量:", len(df))
except Exception as e:
    print(f"数据加载失败，请检查文件名或路径: {e}")
    # 以下为占位逻辑，防止代码因路径问题直接崩溃
    df = pd.DataFrame()

if not df.empty:
    # 自动识别 Friday the 13th
    df['weekday'] = df.index.weekday
    df['day_of_month'] = df.index.day
    df['is_friday_13'] = (df['weekday'] == 4) & (df['day_of_month'] == 13)


    # ==========================================
    # 2. 核心处理逻辑 (去除季节化/趋势化)
    # ==========================================
    def get_unlucky_probability(series, is_positive_unlucky=True):
        """
        直接将原始数值映射为 0-1 之间的“不幸概率”
        """
        series = series.dropna()
        if len(series) == 0: return pd.Series(dtype=float)

        values = series.values
        # 如果数值越大越幸运（如股票回报），则取负值，使其变为越大越不幸
        if not is_positive_unlucky:
            values = -values

        ecdf = ECDF(values)
        return pd.Series(ecdf(values), index=series.index)


    print("\n=== 计算各指标不幸概率 ===")

    # 根据您的截图列名进行映射 (pos=True 表示数值越大越不幸)
    indicator_map = {
        'Occurrence_Count_crime_file_1': True,
        'Occurrence_Count_storm_file_2': True,
        'Trauma_detrended_file_3': True,  # 虽列名含detrended，此处直接使用
        'Detrended_Value_CPI_file_5': True,
        'Total_New_Deaths_file_6': True
    }

    p_cols = []
    for col, pos in indicator_map.items():
        if col in df.columns:
            p_name = f'p_{col}'
            df[p_name] = get_unlucky_probability(df[col], is_positive_unlucky=pos)
            p_cols.append(p_name)
            print(f"已处理指标: {col}")

    # ==========================================
    # 3. 指数合成与统计分析
    # ==========================================
    # 权重分配（可根据研究需求调整）
    df['unlucky_index'] = df[p_cols].mean(axis=1)
    df['unlucky_index'] = df['unlucky_index'].fillna(df['unlucky_index'].median())

    print("\n--- 不幸指数描述性统计 ---")
    print(df.groupby('is_friday_13')['unlucky_index'].describe())

    # t检验
    ft13 = df[df['is_friday_13']]['unlucky_index'].dropna()
    other = df[~df['is_friday_13']]['unlucky_index'].dropna()

    if len(ft13) > 0:
        t_stat, p_val = stats.ttest_ind(ft13, other, equal_var=False)
        print(f"\nT-检验 p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("结论：在 0.05 水平下，13号星期五的影响显著。")
        else:
            print("结论：无统计显著性，13号星期五并不更不幸。")

    # 回归分析
    try:
        # 构建回归矩阵：控制星期几的效应，观察 Friday_13 的系数
        X = pd.get_dummies(df['weekday'], prefix='wd', drop_first=True)
        X['is_friday_13'] = df['is_friday_13'].astype(int)
        X = sm.add_constant(X.astype(float))

        model = sm.OLS(df['unlucky_index'], X).fit()
        print("\n--- 回归分析总结 (关注 is_friday_13 系数) ---")
        print(model.summary().tables[1])
    except Exception as e:
        print(f"回归分析出错: {e}")

    # 可视化
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_friday_13', y='unlucky_index', data=df)
        plt.title('Unlucky Index: Friday the 13th vs Others')
        plt.show()
    except:
        pass

    # 导出结果
    # df.to_csv('analysis_results.csv')
if not df.empty:
    # 1. Ensure all required time features exist for the analysis functions
    # The functions below expect: 'Day_of_Week', 'Year', 'Month', 'Day'
    df['Day_of_Week'] = df.index.weekday  # 0=Monday, 4=Friday
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day

    # Define ts_data for the next step
    ts_data = df
def select_control_group(df):
    """
    在 ts_data 中筛选 Target (13号星期五) 和 Control (同月同年的其他星期五)。
    返回标记好分组的 DataFrame。
    """
    df = df.copy()

    # 1. 基础筛选：只保留星期五
    # 周五的 dayofweek 通常为 4 (Monday=0, Sunday=6)
    fridays = df[df['Day_of_Week'] == 4].copy()

    # 2. 创建匹配键 (Key)
    # 我们只在 "同一个年" 和 "同一个月" 内进行比较
    fridays['Match_Key'] = fridays['Year'].astype(str) + '-' + fridays['Month'].astype(str).str.zfill(2)

    # 3. 标记 Target 和 Control
    # Target: 这一天是 13 号
    fridays['Group_Type'] = np.where(fridays['Day'] == 13, 'Target', 'Control')

    # 4. 关键步骤：过滤无效组
    # 如果某个月有一个 13号星期五，我们需要确保这个月里确实还有其他星期五作为对照。
    # 如果某个月虽然有星期五，但没有13号，那这个月的所有数据对我们的特定假设也是没用的（构不成配对）。

    # 计算每个 Key 下面是否包含 Target
    groups_with_target = fridays.groupby('Match_Key')['Group_Type'].apply(lambda x: (x == 'Target').any())
    valid_keys = groups_with_target[groups_with_target].index

    # 只保留那些“确实包含13号星期五的月份”的数据
    final_dataset = fridays[fridays['Match_Key'].isin(valid_keys)].copy()

    return final_dataset


# 执行筛选
analysis_set = select_control_group(ts_data)

# 查看结果
print(f"原始数据行数: {len(ts_data)}")
print(f"筛选后的分析集行数 (仅含相关月份的周五): {len(analysis_set)}")
print(f"其中目标日 (Target) 数量: {len(analysis_set[analysis_set['Group_Type'] == 'Target'])}")
print(f"其中对照日 (Control) 数量: {len(analysis_set[analysis_set['Group_Type'] == 'Control'])}")


def verify_control_group(df_analysis):
    """
    对筛选后的数据集进行严谨性验证
    """
    print("=== 基准组性质验证报告 ===\n")

    targets = df_analysis[df_analysis['Group_Type'] == 'Target']
    controls = df_analysis[df_analysis['Group_Type'] == 'Control']

    # -------------------------------------------------------
    # 验证 1: 季节性分布 (Month Distribution)
    # 理论上，Month 的分布必须完全一致 (100% 匹配)，否则引入了季节偏差
    # -------------------------------------------------------
    target_month_dist = targets['Month'].value_counts(normalize=True).sort_index()
    control_month_dist = controls['Month'].value_counts(normalize=True).sort_index()

    # 计算差异 (Target 占比 - Control 占比)
    diff = (target_month_dist - control_month_dist).abs().sum()

    print(f"1. 季节性一致性检查 (Sum of Abs Diff): {diff:.4f}")
    if diff < 1e-9:
        print("   [Passed] 完美！目标组和基准组的月份分布完全重合。季节因素已被完全控制。")
    else:
        print("   [Warning] 月份分布不一致，请检查是否某些月份只有 Target 无 Control。")

    # 可视化验证
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 月份分布对比图
    df_analysis.groupby(['Month', 'Group_Type']).size().unstack().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Month Distribution Check')
    axes[0].set_ylabel('Count of Days')

    # -------------------------------------------------------
    # 验证 2: 样本配对比例 (Ratio Check)
    # 检查平均每个“13号星期五”对应了几个“普通星期五”
    # -------------------------------------------------------
    ratios = df_analysis.groupby('Match_Key')['Group_Type'].value_counts().unstack().fillna(0)
    avg_ratio = (ratios['Control'] / ratios['Target']).mean()

    print(f"\n2. 配对比例检查:")
    print(f"   平均每个 13号星期五 匹配到了 {avg_ratio:.2f} 个同月对照日。")
    if avg_ratio >= 3:
        print("   [Good] 样本充足（通常一个月有4-5个周五，除去13号，应剩3-4个）。")
    elif avg_ratio < 1:
        print("   [Critical] 部分目标日没有对照组！")

    # -------------------------------------------------------
    # 验证 3: 有效性检验 (Data Validity)
    # 确保我们没有拿“古代数据”去比“现代数据”
    # -------------------------------------------------------
    # 年份箱线图
    sns.boxplot(x='Group_Type', y='Year', data=df_analysis, ax=axes[1])
    axes[1].set_title('Year Distribution Check')

    # T检验 (检查年份均值是否有显著差异 - 理论上这应该是 0 差异)
    t_stat, p_val = stats.ttest_ind(targets['Year'], controls['Year'])
    print(f"\n3. 时间跨度偏差检查 (Year T-test):")
    print(f"   P-value: {p_val:.4f}")
    if p_val > 0.05:
        print("   [Passed] 两组数据的平均年份没有统计学差异 (P > 0.05)，不存在年代偏差。")
    else:
        print("   [Warning] 年份分布不均匀，可能某几年的13号星期五特别多？")

    plt.tight_layout()
    plt.show()


# 执行验证
verify_control_group(analysis_set)