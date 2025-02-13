import statistics

import numpy as np


def set_fitness_diversity(population):
    fitness_values = [ind.fitness for ind in population]
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
    semantics = [ind.semantic_vector for ind in population]  # List of semantic lists
    pairwise_distances = []

    for i in range(len(semantics)):
        for j in range(i + 1, len(semantics)):
            # Compute semantic distance as the average element-wise difference
            distance = sum(
                abs(semantics[i][k] - semantics[j][k]) for k in range(len(semantics[i]))
            ) / len(semantics[i])
            pairwise_distances.append(distance)

    return sum(pairwise_distances) / len(pairwise_distances) if pairwise_distances else 0
