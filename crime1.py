import pandas as pd

# ================= 配置 =================
INPUT_FILE = 'Debris flow.xlsx'  # 你的原始文件名
DATE_COLUMN = 'BEGIN_DATE'  # 日期所在的列名（根据你之前的截图是 BEGIN_DATE）
OUTPUT_FILE = 'xinzangbing.xlsx'


# ========================================

def count_daily_occurrences():
    print(f"正在读取 {INPUT_FILE}...")

    # 1. 读取 Excel
    try:
        df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 2. 转换日期格式（处理脏数据）
    # errors='coerce' 会将无法识别的文字（比如之前的 irrigation system 描述）转为空值 NaT
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')

    # 3. 统计并清洗
    # 统计前先去掉那些转换失败的空行
    initial_len = len(df)
    df = df.dropna(subset=[DATE_COLUMN])
    print(f"ℹ️ 已过滤 {initial_len - len(df)} 行无效日期数据。")

    # 4. 核心统计：计算每天出现的次数
    # 我们提取出日期部分（去除具体小时分钟），然后进行计数
    daily_stats = df[DATE_COLUMN].dt.date.value_counts().sort_index().reset_index()

    # 重命名列名以便阅读
    daily_stats.columns = ['Date', 'Occurrence_Count']

    # 5. 保存结果
    daily_stats.to_excel(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"✅ 统计完成！结果已保存至: {OUTPUT_FILE}")
    print("预览前 5 天数据:")
    print(daily_stats.head())


if __name__ == "__main__":
    count_daily_occurrences()