import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ────────────────────────── 全局样式设置 ──────────────────────────
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# ────────────────────────── 数据处理 ──────────────────────────
df = pd.read_excel('public_emdat_incl_hist_2026-01-30.xlsx')

for col in ['Start Year', 'Start Month', 'Start Day', 'Total Deaths']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Start Date'] = pd.to_datetime(
    {'year': df['Start Year'], 'month': df['Start Month'].fillna(1), 'day': df['Start Day'].fillna(1)},
    errors='coerce'
)
df = df.dropna(subset=['Start Date'])

daily_deaths = df.set_index('Start Date')['Total Deaths'].resample('D').sum().fillna(0)
daily_counts = df.set_index('Start Date').resample('D').size()

total_days = len(daily_deaths)
zero_days = (daily_deaths == 0).sum()
zero_ratio = zero_days / total_days
max_daily_deaths = daily_deaths.max()
non_zero_deaths = daily_deaths[daily_deaths > 0]

# ────────────────────────── 绘图部分 ──────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=140,
                               gridspec_kw={'width_ratios': [1, 2.2], 'wspace': 0.18})

# --- 左侧子图 (AX1) ---
counts_dist = daily_counts[daily_counts > 0].value_counts().sort_index()
ax1.bar(counts_dist.index, counts_dist.values, color='#455a64', edgecolor='black', alpha=0.8)
ax1.set_title('Frequency of Daily Incident Counts', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Number of Incidents per Day', fontsize=12)
ax1.set_ylabel('Frequency (Days)', fontsize=12)
ax1.grid(True, axis='y', ls='--', alpha=0.4)

# --- 右侧子图 (AX2) ---
bins_log = np.logspace(np.log10(1), np.log10(max_daily_deaths + 1), num=45)
n, bin_edges, patches = ax2.hist(
    non_zero_deaths, bins=bins_log, color='#d32f2f', edgecolor='black', alpha=0.85, linewidth=0.5
)

# ────────────────────────── 重点：向右下移动注释 ──────────────────────────
# x 设置为 2000 (向右移), y 设置为 max(n) * 0.05 (向下移)
ax2.text(
    5, max(n) * 0.01,
    f'Zero-Death Days: {zero_days:,}\n'
    f'Proportion: {zero_ratio:.1%}\n'
    f'(A Signature of Zero-Inflation)\n'
    f'Total Observed Days: {total_days:,}\n'
    f'Max Daily Deaths: {int(max_daily_deaths):,}',
    fontsize=10, fontweight='bold',
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#b71c1c', boxstyle='round,pad=0.5'),
    va='bottom', ha='left' # 改为从底部对齐
)

# 重尾注释也同步微调位置，避免重叠
'''
ax2.text(
    max_daily_deaths * 0.4, max(n) * 0.015,
    'Heavy Tail: Extreme Events\nDominating Total Impact',
    fontsize=9, color='#7b1fa2', fontweight='semibold',
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'),
    va='bottom'
)
'''

# 右图坐标轴设置
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Distribution of Daily Total Deaths (Log-Log)', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Daily Total Deaths (log scale)', fontsize=12)
ax2.set_ylabel('Frequency of Days (log scale)', fontsize=12)

xticks = [1, 10, 100, 1000, 10000, 100000, 1000000, 3000000]
ax2.set_xticks(xticks)
ax2.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}' if x < 1e6 else f'{x/1e6:.1f}M'))
ax2.grid(True, which="both", ls="--", alpha=0.3)

# ────────────────────────── 布局对齐 ──────────────────────────
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.96)

output_path = 'analysis_bottom_right.png'
plt.savefig(output_path, dpi=400, bbox_inches='tight')
plt.show()