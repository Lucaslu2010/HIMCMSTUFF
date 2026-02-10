'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 基础参数
# -----------------------------
cities = {
    "sh": {"N0": 806.67, "K": 3000, "r": 0.01, "P0": 0.098},
    "sz": {"N0": 419.67, "K": 2000, "r": 0.01, "P0": 0.087},
    "nyc": {"N0": 277.67, "K": 2000, "r": 0.01, "P0": 0.099},
}

g = 0.001       # 适配比例年增长率
R = 1.0         # 留存率
years = np.arange(0, 16)

# -----------------------------
# 逻辑增长函数
# -----------------------------
def logistic_growth(t, N0, K, r):
    return K / (1 + (K - N0) / N0 * np.exp(-r * t))

# -----------------------------
# 任务 1 表格计算
# -----------------------------
records = []

for city, p in cities.items():
    for t in [5, 10, 15]:
        N_t = logistic_growth(t, p["N0"], p["K"], p["r"])
        P_t = p["P0"] + g * t
        pets = N_t * P_t * R

        records.append([city, 2024 + t, N_t, P_t, pets])

task1_table = pd.DataFrame(
    records,
    columns=["chenshi", "nianfen", "jiating number wan", "pet%", "PetNumber wan"]
)

task1_table

r_values = [0.005, 0.01, 0.015]
records_r = []

for r in r_values:
    N_2039 = logistic_growth(15, cities["sh"]["N0"], cities["sh"]["K"], r)
    P_2039 = cities["sh"]["P0"] + g * 15
    pets = N_2039 * P_2039
    records_r.append([r, pets])

task2_r_table = pd.DataFrame(
    records_r,
    columns=["内在增长率 r", "2039 年宠物数量（万）"]
)

task2_r_table


K_values = [1600, 2000, 2400]
records_K = []

for K in K_values:
    N_2039 = logistic_growth(15, cities["sz"]["N0"], K, cities["sz"]["r"])
    P_2039 = cities["sz"]["P0"] + g * 15
    pets = N_2039 * P_2039
    records_K.append([K, pets])

task2_K_table = pd.DataFrame(
    records_K,
    columns=["承载量 K（万）", "2039 年宠物数量（万）"]
)

task2_K_table


plt.figure(figsize=(8, 5))
plt.rcParams['font.family'] = 'Times New Roman'

for city, p in cities.items():
    pets_series = []
    for t in years:
        N_t = logistic_growth(t, p["N0"], p["K"], p["r"])
        P_t = p["P0"] + g * t
        pets_series.append(N_t * P_t)

    plt.plot(2024 + years, pets_series, label=city)

plt.xlabel("nianfen")
plt.ylabel("PetNumber Wan")
plt.title("2024–2039 pet change")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Model functions
# -----------------------------
def logistic_growth(t, N0, K, r):
    return K / (1 + (K - N0) / N0 * np.exp(-r * t))

# -----------------------------
# Parameters
# -----------------------------
g = 0.001
t_target = 15

# Shanghai parameters
N0_sh = 806.67
K_sh = 3000
r_sh = 0.01
P0_sh = 0.098

# Shenzhen parameters
N0_sz = 419.67
K_sz = 2000
r_sz = 0.01
P0_sz = 0.087

# -----------------------------
# Sensitivity to r (Shanghai)
# -----------------------------
r_values = np.array([0.005, 0.01, 0.015])
pets_r = []

for r in r_values:
    N_t = logistic_growth(t_target, N0_sh, K_sh, r)
    P_t = P0_sh + g * t_target
    pets_r.append(N_t * P_t)

pets_r = np.array(pets_r)

# -----------------------------
# Sensitivity to K (Shenzhen)
# -----------------------------
K_values = np.array([1600, 2000, 2400])
pets_K = []

for K in K_values:
    N_t = logistic_growth(t_target, N0_sz, K, r_sz)
    P_t = P0_sz + g * t_target
    pets_K.append(N_t * P_t)

pets_K = np.array(pets_K)

# -----------------------------
# Font settings
# -----------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(r_values, pets_r, marker='o', linestyle='-', label='Sensitivity to r (Shanghai)')
plt.plot(K_values, pets_K, marker='s', linestyle='-', label='Sensitivity to K (Shenzhen)')

plt.xlabel("Parameter Value")
plt.ylabel("Pet Quantity in 2039")
plt.title("Sensitivity Analysis Comparison")
plt.legend()
plt.grid(True)

plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Model functions
# -----------------------------
def logistic_growth(t, N0, K, r):
    return K / (1 + (K - N0) / N0 * np.exp(-r * t))

def adaptive_growth_rate(N, K, g0):
    return g0 * (1 - N / K)

# -----------------------------
# Parameters
# -----------------------------
g = 0.001
years = np.array([5, 10, 15])

# Shanghai parameters
N0_sh = 806.67
K_sh = 3000
r_sh = 0.01
P0_sh = 0.098

# Shenzhen parameters
N0_sz = 419.67
K_sz = 2000
r_sz = 0.01
P0_sz = 0.087

# -----------------------------
# Task 3.1: Declining retention rate (Shanghai)
# -----------------------------
pets_retention = []

for t in years:
    N_t = logistic_growth(t, N0_sh, K_sh, r_sh)
    P_t = P0_sh + g * t
    R_t = 1 - 0.005 * t
    pets_retention.append(N_t * P_t * R_t)

pets_retention = np.array(pets_retention)

# -----------------------------
# Task 3.2: Adaptive adoption growth (Shenzhen)
# -----------------------------
pets_adaptive = []

for t in years:
    N_t = logistic_growth(t, N0_sz, K_sz, r_sz)
    g_eff = adaptive_growth_rate(N_t, K_sz, g)
    P_t = P0_sz + g_eff * t
    pets_adaptive.append(N_t * P_t)

pets_adaptive = np.array(pets_adaptive)

# -----------------------------
# Font settings
# -----------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(2024 + years, pets_retention, marker='o', linestyle='-',
         label='xiajiang Shanghai)')
plt.plot(2024 + years, pets_adaptive, marker='s', linestyle='-',
         label='gaibian lingyang Shenzhen)')

plt.xlabel("Year")
plt.ylabel("Pet Quantity")
plt.title("Model Extensions Comparison")
plt.legend()
plt.grid(True)

plt.show()

