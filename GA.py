import random
import math


W, H = 100, 100
r_obs = 10
obs_x, obs_y = 50, 50
population_size = 30
binary_length = 16
mutation_rate = 0.01
max_generations = 100
epsilon = 0.000001
elite_size = 2

def random_individual():
    x = random.uniform(0, W)
    y = random.uniform(0, H)
    return x, y


def encode(x, y):
    x_bin = format(int(x / W * (2 ** binary_length - 1)), f'0{binary_length}b')
    y_bin = format(int(y / H * (2 ** binary_length - 1)), f'0{binary_length}b')
    return x_bin + y_bin

def decode(binary):
    x_bin = binary[:binary_length]
    y_bin = binary[binary_length:]
    x = int(x_bin, 2) / (2 ** binary_length - 1) * W
    y = int(y_bin, 2) / (2 ** binary_length - 1) * H
    return x, y

def fitness(x, y):
    d_obs = math.hypot(x - obs_x, y - obs_y) - r_obs
    d_edges = min(x, W - x, y, H - y)
    return max(min(d_obs, d_edges), 0)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(binary):
    return ''.join(
        bit if random.random() > mutation_rate else str(1 - int(bit))
        for bit in binary
    )

population = [encode(*random_individual()) for i in range(population_size)]

max_R_list = []
cnt = 0
for generation in range(max_generations):
    decoded = [decode(ind) for ind in population]
    scored = [(ind, fitness(x, y)) for ind, (x, y) in zip(population, decoded)]
    scored.sort(key=lambda x: x[1], reverse=True)

    max_R = scored[0][1]
    max_R_list.append(max_R)

    if cnt == 1000:
        break
    cnt += 1
    '''
        if generation > 1 and abs(max_R_list[-1] - max_R_list[-2]) < epsilon:
            break
    '''
    new_population = [ind for ind, i in scored[:elite_size]]

    fitness_values = [fit for i, fit in scored]
    total_fitness = sum(fitness_values)

    def select_parent():
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind, fit in scored:
            current += fit
            if current >= pick:
                return ind
        return scored[-1][0]

    while len(new_population) < population_size:
        p1 = select_parent()
        p2 = select_parent()
        c1, c2 = crossover(p1, p2)
        new_population.append(mutate(c1))
        if len(new_population) < population_size:
            new_population.append(mutate(c2))

    population = new_population[:population_size]

best_bin = scored[0][0]
best_x, best_y = decode(best_bin)
best_R = scored[0][1]

print(f"Best center: ({best_x:.2f}, {best_y:.2f}) with radius R = {best_R:.2f}")
#print(f"Max R over generations: {max_R_list}")
for a in max_R_list:
    print(a, end = '\n')
