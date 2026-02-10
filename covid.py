import pandas as pd

# ================= 配置 =================
INPUT_FILE = 'compact.csv'  # 原始文件名
OUTPUT_FILE = 'daily_global_deaths.xlsx'


# ========================================

def aggregate_daily_deaths():
    print(f"正在读取 {INPUT_FILE}...")

    # 1. 读取 Excel
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 2. 清洗列名和日期
    # 去除列名两端的空格 (防止 'Date ' 或 ' new_deaths' 导致报错)
    df.columns = df.columns.str.strip()

    # 假设日期列名为 'date'，死亡人数列名为 'new_deaths'
    # 如果你的列名不同，请在此处修改
    date_col = 'date'
    value_col = 'new_deaths'

    # 强制转换日期格式，并将无法识别的文字转为空值 (NaT)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # 删除日期为空或死亡人数为空的行
    df = df.dropna(subset=[date_col, value_col])

    # 3. 核心步骤：按日期累加
    # .dt.date 确保我们忽略具体的时间（小时/分钟），只按“天”统计
    # .sum() 会将同一天内所有国家/地区的 new_deaths 加在一起
    print("正在进行全球每日死亡人数累加...")
    daily_total = df.groupby(df[date_col].dt.date)[value_col].sum().reset_index()

    # 4. 排序并保存
    daily_total = daily_total.sort_values(by=date_col)

    # 重命名列名以便更清晰
    daily_total.columns = ['Date', 'Total_New_Deaths']

    daily_total.to_excel(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"✅ 处理完成！")
    print(f"结果预览:\n", daily_total.head())
    print(f"结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    aggregate_daily_deaths()