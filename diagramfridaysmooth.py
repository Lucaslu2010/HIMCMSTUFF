import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------- 设置部分 -------------------
excel_file = "unhappy_index_result.xlsx"    # ← 改成你的实际文件名
sheet_name = "Sheet1"                       # 如有需要请修改工作表名

# ------------------- 读取数据 -------------------
df = pd.read_excel(
    excel_file,
    sheet_name=sheet_name,
    usecols=[0, 7],                    # 只读第一列（日期） + 第八列（数值）
    header=0                           # 假设第一行是标题
)

# 第一列转成 datetime
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

# 移除无效日期行
df = df.dropna(subset=[df.columns[0]])

# 按日期排序（保险起见）
df = df.sort_values(by=df.columns[0])

# 重命名方便阅读
df = df.rename(columns={
    df.columns[0]: '日期',
    df.columns[1]: '数值'
})

# ─────────────── 核心：按每周五重采样 ───────────────
df.set_index('日期', inplace=True)

# 重采样到每周五，取该周最后的值
df_weekly_friday = df.resample('W-FRI').last()

# 移除全为空的周
df_weekly_friday = df_weekly_friday.dropna()

# 索引变回普通列
df_weekly_friday = df_weekly_friday.reset_index()

# ─────────────── 平滑计算（这里用 EWMA，你也可以换成 rolling） ───────────────
span_value = 3                 # ← 可调整：越大越平滑（建议6~20）
df_weekly_friday['数值_平滑'] = df_weekly_friday['数值'].ewm(
    span=span_value,
    adjust=True
).mean()
plt.rcParams['font.family'] = 'Times New Roman'

# ─────────────── 绘图 ───────────────
plt.figure(figsize=(14, 4.5))   # 稍高一点，更好看文字和图例

# 1. 原始数据（很淡的线 + 点）
plt.plot(
    df_weekly_friday['日期'],
    df_weekly_friday['数值'],
    color='#bbbbbb',
    linewidth=0.9,
    alpha=0.6,
    zorder=1
)

plt.scatter(
    df_weekly_friday['日期'],
    df_weekly_friday['数值'],
    color='#888888',
    s=25,                   # 点的大小
    alpha=0.45,
    edgecolor='none',
    zorder=2
)

# 2. 平滑曲线（主要展示）
plt.plot(
    df_weekly_friday['日期'],
    df_weekly_friday['数值_平滑'],
    color='#0066cc',
    linewidth=2.5,
    label=f'smoothed (span={span_value})',
    zorder=3
)

# 日期刻度设置
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=18))  # 自动控制刻度数量

plt.xticks(rotation=35, ha='right')

# 美化
plt.title("CMI Changing Through Time — Weekly (Smoothed)", fontsize=14, pad=15)
plt.xlabel("Date (Weekly Friday)", fontsize=11)
plt.ylabel("Unlucky Index", fontsize=11)

plt.grid(True, linestyle='--', alpha=0.22, zorder=0)

plt.legend(loc='upper right', fontsize=10, framealpha=0.92)

plt.tight_layout()
plt.savefig("friday_unhappy_index_smoothed.png", dpi=300, bbox_inches='tight')
# 显示
plt.show()

# 如需保存高清版，取消注释下面这行
