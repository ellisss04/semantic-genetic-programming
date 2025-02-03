import numpy as np
from collections import Counter


def track_fitness_diversity(population):
    fitness_values = [ind.fitness for ind in population]
    return np.std(fitness_values)  # Standard deviation of fitness


def track_genotypic_diversity(population):
    depths = [ind.tree.get_depth() for ind in population]
    return np.std(depths)  # Standard deviation of tree depths


def track_unique_individuals(population):
    unique_individuals = []
    for ind in population:
        if str(ind) not in unique_individuals:
            unique_individuals.append(str(ind))
    return len(unique_individuals) / len(population)


def track_shannon_entropy(population):
    fitness_values = [ind.fitness for ind in population]
    freq = Counter(fitness_values)
    total = sum(freq.values())
    probabilities = [count / total for count in freq.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)


