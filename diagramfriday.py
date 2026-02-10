import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------- 设置部分 -------------------
excel_file = "unhappy_index_result.xlsx"          # ← 改成你的实际文件名
sheet_name = "Sheet1"                  # 如有需要请修改工作表名

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

# 重命名方便阅读（可选）
df = df.rename(columns={
    df.columns[0]: '日期',
    df.columns[1]: '数值'
})

# ─────────────── 核心：按每周五重采样 ───────────────
# 先设日期为索引
df.set_index('日期', inplace=True)

# 重采样到每周五，取该周最后的值
df_weekly_friday = df.resample('W-FRI').last()

# 移除全为空的周（如果有）
df_weekly_friday = df_weekly_friday.dropna()

# 如果你想要索引变回普通列，方便后续使用
df_weekly_friday = df_weekly_friday.reset_index()

# ─────────────── 绘图 ───────────────
plt.figure(figsize=(14, 3.2))   # 细长比例，可自行调整宽高

plt.plot(
    df_weekly_friday['日期'],
    df_weekly_friday['数值'],
    color='#0066cc',
    linewidth=1.6,
    marker='o',
    markersize=5,
    markerfacecolor='white',
    markeredgewidth=1.5,
    alpha=0.95
)

# 日期格式与刻度控制
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=15))  # 自动刻度，最多约15个

plt.xticks(rotation=35, ha='right')

plt.title("fridaygraph", fontsize=13, pad=12)
plt.xlabel("weeks）", fontsize=11)
plt.ylabel("data", fontsize=11)

plt.grid(True, linestyle='--', alpha=0.25)

plt.tight_layout()
plt.show()

# 如需保存高清图，可取消注释下面这行
# plt.savefig("每周五趋势_重采样版.png", dpi=300, bbox_inches='tight')