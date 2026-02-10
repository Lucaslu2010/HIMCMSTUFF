import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------- 设置部分 -------------------
excel_file = "unhappy_index_result.xlsx"          # ← 改成你的文件名
sheet_name = "Sheet1"                  # 如有需要请修改工作表名
date_column = 0                        # 第一列（从0开始计数）
value_column = 7                       # 第八列（从0开始计数）

# ------------------- 读取数据 -------------------
df = pd.read_excel(
    excel_file,
    sheet_name=sheet_name,
    usecols=[date_column, value_column],   # 只读取这两列，速度更快
    header=0                               # 假设第一行是标题行
)

# 保证第一列是日期格式
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

# 删除无法转换为日期的行（如果有）
df = df.dropna(subset=[df.columns[0]])

# 按日期排序（以防Excel里没排好序）
df = df.sort_values(by=df.columns[0])

# ------------------- 绘图 -------------------
# 创建细长的画布（宽:高 ≈ 4:1 或更大）
plt.figure(figsize=(12, 3))     # 你可以改成 (14, 2.5)、(16, 3) 等更细长比例

plt.plot(
    df.iloc[:, 0],              # x轴：日期
    df.iloc[:, 1],              # y轴：第八列数据
    color='#0066cc',            # 深蓝色（可改）
    linewidth=1.4,              # 线条粗细
    marker='o',                 # 数据点用小圆点（可选，可删除这行）
    markersize=4,               # 小圆点大小
    alpha=0.9
)

# 美化日期刻度
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))  # 自动控制刻度数量

# 旋转日期标签，防止重叠
plt.xticks(rotation=30, ha='right')

# 标题和轴标签（可根据需要修改或删除）
plt.title("data diagram", fontsize=13, pad=12)
plt.xlabel("date", fontsize=11)
plt.ylabel("distancing index", fontsize=11)

# 添加网格（可选）
plt.grid(True, linestyle='--', alpha=0.3)

# 紧凑布局，避免边缘被裁切
plt.tight_layout()

# 显示图像
plt.show()

# 如果想要保存为高清图片，可以取消下面这行的注释
# plt.savefig("趋势图_细长版.png", dpi=300, bbox_inches='tight')