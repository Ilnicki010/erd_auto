import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pygad
from itertools import combinations
from shapely.geometry import LineString

# Define the ERD diagram as a list of entities and their connections
entities = ["entity" + str(i) for i in range(1, 7)]
connections = [('entity1', 'entity2'), ('entity1', 'entity3'), ('entity2', 'entity4'), ('entity3', 'entity4'),
               ('entity4', 'entity5')]

GRID_MULTIPLIER = 3
GRID_SIZE = len(entities) * GRID_MULTIPLIER


def plot_solution(solution):
    n = int(len(solution) / 2)
    fig, ax = plt.subplots()
    for i in range(n):
        label = entities[i]
        x, y = solution[2 * i], solution[2 * i + 1]
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, color="black"))
        ax.text(x + 0.5, y + 0.5, label, ha='center', va='center', color="red", size=16)
    # for conn in connections:
    #     i, j = entities.index(conn[0]), entities.index(conn[1])
    #     x1, y1, x2, y2 = solution[2 * i] + 0.5, solution[2 * i + 1] + 0.5, solution[2 * j] + 0.5, solution[
    #         2 * j + 1] + 0.5
    #     ax.plot([x1, x2], [y1, y2], 'k-')
    ax.set_xlim(0, GRID_SIZE + 1)
    ax.set_ylim(0, GRID_SIZE + 1)
    plt.grid(True)
    plt.show()


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_related_entities(target_entity: str) -> list[str]:
    related_entities = []
    for conn in connections:
        if conn[0] == target_entity:
            related_entities.append(conn[1])
    return related_entities


def fitness_function(solution, solution_idx):
    center = (GRID_SIZE / 2, GRID_SIZE / 2)
    n = len(solution)

    all_points_from_center = []
    all_dist_from_other = []
    for i in range(0, n - 1, 2):
        x1, y1 = solution[i], solution[i + 1]
        dist_from_center = calculate_distance(x1, y1, center[0], center[1])
        all_points_from_center.append(dist_from_center)

        # check if position is the same
        if x1 in [x for x in solution[i + 1:] if x % 2 == 0] and y1 in [x for x in solution[i + 1:] if x % 2 != 0]:
            return -99999

        for j in range(i + 2, n - 1, 2):
            x2, y2 = solution[j], solution[j + 1]

            dist_from_other = calculate_distance(x1, y1, x2, y2)
            all_dist_from_other.append(dist_from_other)

    dist_matrix = [[solution[i], solution[i + 1]] for i in range(0, len(solution) - 1, 2)]
    lines = [LineString((p1, p2)) for p1, p2 in combinations(dist_matrix, 2)]
    num_crossovers = sum(1 for i, l1 in enumerate(lines) for l2 in lines[i + 1:] if l1.intersects(l2))

    return -np.mean(all_points_from_center) * 1.5 + np.mean(all_dist_from_other) - num_crossovers ** 2


def generate_initial_population(n: int):
    return [random.sample(range(GRID_SIZE + 1), len(entities) * 2) for _ in range(n)]


ga_instance = pygad.GA(gene_space=list(range(GRID_SIZE + 1)),
                       initial_population=generate_initial_population(40),
                       crossover_type='single_point',
                       num_generations=300,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       num_genes=len(entities) * 2,  # every table has x and y
                       parent_selection_type="sss",
                       keep_parents=2,
                       mutation_type='random',
                       mutation_percent_genes=40
                       )
ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()

print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")

ga_instance.plot_fitness()
plot_solution(solution)
