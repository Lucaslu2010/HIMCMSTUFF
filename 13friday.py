import pandas as pd
import numpy as np
from scipy.stats import binomtest

# =====================================================
# 0. FILE PATH
# =====================================================

EXCEL_FILE = "public_emdat_incl_hist_2026-01-30.xlsx"   # <-- change if needed

# =====================================================
# 1. LOAD EXCEL DATA
# =====================================================

df = pd.read_excel(EXCEL_FILE)

# =====================================================
# 2. EXTRACT START DATE COMPONENTS
# Excel columns:
# Z  = year  -> index 25
# AA = month -> index 26
# AB = day   -> index 27
# =====================================================

df["start_year"] = pd.to_numeric(df.iloc[:, 25], errors="coerce")
df["start_month"] = pd.to_numeric(df.iloc[:, 26], errors="coerce")
df["start_day"] = pd.to_numeric(df.iloc[:, 27], errors="coerce")

# Drop rows with missing date components
df = df.dropna(subset=["start_year", "start_month", "start_day"])

# Convert to integers
df["start_year"] = df["start_year"].astype(int)
df["start_month"] = df["start_month"].astype(int)
df["start_day"] = df["start_day"].astype(int)

# =====================================================
# 3. CONSTRUCT START DATE (ROBUST)
# =====================================================

df["start_date"] = df.apply(
    lambda r: pd.Timestamp(
        year=r["start_year"],
        month=r["start_month"],
        day=r["start_day"]
    ),
    axis=1
)

# =====================================================
# 4. IDENTIFY FRIDAY THE 13TH
# =====================================================

# Monday=0, ..., Friday=4
df["is_friday_13"] = (
    (df["start_date"].dt.day == 13) &
    (df["start_date"].dt.weekday == 4)
)

# =====================================================
# 5. EVENT-FREQUENCY ANALYSIS (UNWEIGHTED)
# =====================================================

observed_events = int(df["is_friday_13"].sum())
total_events = len(df)

# Calendar-based expected probability
start = df["start_date"].min()
end = df["start_date"].max()
all_days = pd.date_range(start, end, freq="D")

calendar_friday_13 = sum(
    (d.day == 13 and d.weekday() == 4)
    for d in all_days
)

expected_p = calendar_friday_13 / len(all_days)

binom_result = binomtest(
    k=observed_events,
    n=total_events,
    p=expected_p,
    alternative="greater"
)

# =====================================================
# 6. SEVERITY-WEIGHTED ANALYSIS (TOTAL DEATHS)
# =====================================================

# Ensure column exists
if "Total Deaths" not in df.columns:
    raise ValueError("Column 'Total Deaths' not found in the Excel file.")

df["deaths"] = pd.to_numeric(df["Total Deaths"], errors="coerce").fillna(0)

total_deaths = df["deaths"].sum()
fri13_deaths = df.loc[df["is_friday_13"], "deaths"].sum()

observed_death_share = fri13_deaths / total_deaths

# Expected share equals calendar probability
expected_death_share = expected_p

# =====================================================
# 7. BOOTSTRAP INFERENCE (WEIGHTED)
# =====================================================

np.random.seed(42)
B = 5000
boot_stats = []

for _ in range(B):
    sample = df.sample(n=len(df), replace=True)
    stat = (
        sample.loc[sample["is_friday_13"], "deaths"].sum()
        / sample["deaths"].sum()
    )
    boot_stats.append(stat)

boot_stats = np.array(boot_stats)
bootstrap_p_value = np.mean(boot_stats >= observed_death_share)

# =====================================================
# 8. OUTPUT RESULTS
# =====================================================

print("=" * 70)
print("FRIDAY THE 13TH DISASTER ANALYSIS")
print("=" * 70)

print("\n--- Event Frequency (Unweighted) ---")
print(f"Total events: {total_events}")
print(f"Friday-13 events: {observed_events}")
print(f"Observed probability: {observed_events / total_events:.6f}")
print(f"Expected probability (calendar): {expected_p:.6f}")
print(f"Binomial test p-value: {binom_result.pvalue:.6f}")

if binom_result.pvalue < 0.05:
    print("Conclusion: Events are significantly MORE frequent on Friday the 13th.")
else:
    print("Conclusion: No evidence of higher event frequency on Friday the 13th.")

print("\n--- Severity-Weighted (Total Deaths) ---")
print(f"Total deaths: {int(total_deaths):,}")
print(f"Deaths on Friday 13th: {int(fri13_deaths):,}")
print(f"Observed death share: {observed_death_share:.6f}")
print(f"Expected death share (calendar): {expected_death_share:.6f}")
print(f"Bootstrap p-value: {bootstrap_p_value:.6f}")

if bootstrap_p_value < 0.05:
    print("Conclusion: Human cost is disproportionately higher on Friday the 13th.")
else:
    print("Conclusion: No evidence of higher human cost on Friday the 13th.")

print("=" * 70)
