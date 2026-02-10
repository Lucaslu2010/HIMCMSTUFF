import pandas as pd
from datetime import datetime, timedelta
import calendar
import os


def normalize_date_with_time(file_path, output_path):
    try:
        # 1. 读取 Excel 文件
        df = pd.read_excel(file_path)
        print(f"成功读取文件: {file_path}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 默认处理第一列，如果你的日期列有标题，可以改为 df['列名']
    date_column = df.iloc[:, 0]

    corrected_dates = []
    cumulative_offset = 0
    new_offsets = []
    total_offsets = []

    for idx, raw_val in enumerate(date_column):
        try:
            # --- 核心修改：剔除时间部分 ---
            # 将输入转为字符串，按空格切分，只取第一部分 (日期部分)
            # 例如 "2000-02-32 10:00" -> "2000-02-32"
            date_part = str(raw_val).strip().split(' ')[0]

            # 如果日期里用的是 / 而不是 -，统一替换成 -
            date_part = date_part.replace('/', '-')

            # 按 '-' 分割年、月、日
            parts = date_part.split('-')
            if len(parts) != 3:
                raise ValueError(f"日期格式无法解析: {date_part}")

            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])

            # --- 以下是之前的偏差修正逻辑 ---
            # 1. 获取该月实际最大天数
            _, last_day_of_month = calendar.monthrange(y, m)

            # 2. 计算本行产生的原始偏差
            current_row_excess = 0
            if d > last_day_of_month:
                current_row_excess = d - last_day_of_month

            # 3. 应用累积偏移并计算真实日期
            base_date = datetime(y, m, 1)
            actual_date = base_date + timedelta(days=(d - 1) + cumulative_offset)

            # 4. 更新偏差和记录结果
            new_offsets.append(current_row_excess)
            cumulative_offset += current_row_excess

            total_offsets.append(cumulative_offset)
            corrected_dates.append(actual_date.strftime('%Y-%m-%d'))

        except Exception as e:
            print(f"第 {idx + 1} 行数据处理出错 ({raw_val}): {e}")
            corrected_dates.append("解析失败")
            new_offsets.append(0)
            total_offsets.append(cumulative_offset)

    # 将结果保存到原表格的新列中
    df['清洗后日期'] = corrected_dates
    df['本行产生的偏差天数'] = new_offsets
    df['当前总累积偏差值'] = total_offsets

    # 保存
    try:
        df.to_excel(output_path, index=False)
        print(f"\n--- 处理完成 ---")
        print(f"修正后的结果已导出至: {output_path}")
    except Exception as e:
        print(f"导出失败，请检查 Excel 文件是否已关闭: {e}")


if __name__ == "__main__":
    # 配置输入输出文件名
    input_file = 'xiuzhengriqi.xlsx'
    output_file = 'xiuzhengriqi_fixed.xlsx'

    if os.path.exists(input_file):
        normalize_date_with_time(input_file, output_file)
    else:
        print(f"错误：在当前目录下找不到文件 {input_file}")