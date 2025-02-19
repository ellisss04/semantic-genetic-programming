import math
from GP.utils import get_pairwise_distance
import statistics

import numpy as np


def set_fitness_diversity(population):
    fitness_values = [ind.fitness for ind in population]
    for i, val in enumerate(fitness_values):
        fitness_values[i] = abs(val)
    fitness_values = np.log(fitness_values)
    return statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0


def track_genotypic_diversity(population):
    depths = [ind.tree.get_depth() for ind in population]
    return np.std(depths)  # Standard deviation of tree depths


def track_unique_individuals(population):
    unique_individuals = []
    for ind in population:
        if str(ind) not in unique_individuals:
            unique_individuals.append(str(ind))
    return len(unique_individuals) / len(population)


def set_semantic_diversity(population):
    """
    Computes semantic diversity as the average pairwise distance between
    individuals' semantics across the dataset.

    Args:
        population (list): List of individuals in the population.

    Returns:
        float: Average pairwise semantic diversity.
    """
    pairwise_matrix = get_pairwise_distance(population)

    # Extract the upper triangle of the matrix without the diagonal
    n = len(population)
    upper_triangle = pairwise_matrix[np.triu_indices(n, k=1)]

    # Compute the average pairwise semantic diversity
    return np.mean(upper_triangle) if len(upper_triangle) > 0 else 0

