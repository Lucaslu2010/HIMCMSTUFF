import numpy as np
import pandas as pd
import random
from collections import defaultdict

np.random.seed(42)
num_boxes = 50
box_ids = [f"药品{i+1}" for i in range(num_boxes)]
lengths = np.random.uniform(80, 160, num_boxes).round(1)
widths = np.random.uniform(20, 80, num_boxes).round(1)
heights = np.random.uniform(10, 60, num_boxes).round(1)
不  \
df_boxes = pd.DataFrame({
    "药品ID": box_ids,
    "长度(mm)": lengths,
    "宽度(mm)": widths,
    "高度(mm)": heights
})
box_widths = df_boxes["宽度(mm)"].values + 2.0


POP_SIZE = 100
N_GENERATIONS = 200
MUTATION_RATE = 0.1
ELITE_SIZE = 5
n_boxes = len(box_widths)

def init_population(): 
    return [np.random.randint(0, n_boxes, size=n_boxes) for _ in range(POP_SIZE)]

def decode(individual):
    slots = defaultdict(list)
    for i, slot_type in enumerate(individual):
        slots[slot_type].append(box_widths[i])
    slot_widths = {k: max(v) for k, v in slots.items()}
    return slot_widths

def fitness(individual):
    slot_widths = decode(individual)
    total_waste = 0
    for i, slot_type in enumerate(individual):
        waste = slot_widths[slot_type] - box_widths[i]
        total_waste += waste
    return len(slot_widths) * 100 + total_waste  # 权重100控制种类优先

def select(population, scores):
    selected = []
    for _ in range(POP_SIZE):
        i, j = np.random.randint(0, POP_SIZE, 2)
        selected.append(population[i] if scores[i] < scores[j] else population[j])
    return selected

def crossover(p1, p2):
    point = np.random.randint(1, n_boxes - 1)
    return np.concatenate((p1[:point], p2[point:]))

def mutate(individual):
    for i in range(n_boxes):
        if np.random.rand() < MUTATION_RATE:
            individual[i] = np.random.randint(0, n_boxes)
    return individual


population = init_population()
for generation in range(N_GENERATIONS):
    scores = [fitness(ind) for ind in population]
    ranked = sorted(zip(scores, population), key=lambda x: x[0])
    elites = [x[1] for x in ranked[:ELITE_SIZE]]
    selected = select(population, scores)
    next_gen = elites[:]
    while len(next_gen) < POP_SIZE:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        next_gen.append(child)
    population = next_gen


best_individual = ranked[0][1]
best_slot_widths = decode(best_individual)
slot_assignments = best_individual
slot_width_dict = {k: round(v, 1) for k, v in best_slot_widths.items()}

df_boxes["槽型编号"] = slot_assignments
df_boxes["分配槽宽(mm)"] = df_boxes["槽型编号"].map(slot_width_dict)

print(f"最优药槽种类数：{len(slot_width_dict)}")
print("药槽宽度列表（单位：mm）：")
print(sorted(slot_width_dict.values()))
