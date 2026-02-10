import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------- 設置部分 -------------------
excel_file = "unhappy_index_result.xlsx"   # ← 改成實際檔名
sheet_name = "Sheet1"                  # 如有需要請修改

# ------------------- 讀取數據 -------------------
df = pd.read_excel(
    excel_file,
    sheet_name=sheet_name,
    usecols=[0, 7],
    header=0
)

# 取得實際列名（更安全）
date_col = df.columns[0]   # 應該是 'Date' 或類似
value_col = df.columns[1]

# 關鍵：使用正確的格式 '%m/%d/%y'
df[date_col] = pd.to_datetime(
    df[date_col],
    format='%m/%d/%y',          # ← 這裡改成這個！
    errors='coerce'
)

# 刪除轉換失敗的行（如果有）
df = df.dropna(subset=[date_col])

# 排序
df = df.sort_values(by=date_col)

# 檢查是否成功（建議先跑這幾行看輸出）
print("日期列 dtype：", df[date_col].dtype)                # 應為 datetime64[ns]
print("\n前 8 行日期（轉換後）：")
print(df[date_col].head(8).dt.strftime('%Y-%m-%d').to_list())
print("\n總行數：", len(df))
print("其中是 13 號的數量：", (df[date_col].dt.day == 13).sum())

# ─────────────── 只保留每月 13 號 ───────────────
df_13th = df[df[date_col].dt.day == 13].copy()

if df_13th.empty:
    print("沒有找到任何 13 號的資料，請檢查資料範圍是否包含 13 日")
else:
    print(f"找到 {len(df_13th)} 筆 13 號資料")

# ─────────────── 繪圖 ───────────────
plt.figure(figsize=(14, 3.5))

plt.plot(
    df_13th[date_col],
    df_13th[value_col],
    color='#0066cc',
    linewidth=1.6,
    marker='o',
    markersize=6,
    markerfacecolor='white',
    markeredgewidth=1.8,
    alpha=0.95
)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每 3 個月一個刻度

plt.xticks(rotation=35, ha='right')

plt.title("CMI changing through time", fontsize=13, pad=12)
plt.xlabel("13th every month", fontsize=11)
plt.ylabel("unlucky index", fontsize=11)

plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()