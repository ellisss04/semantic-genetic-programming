import math
import random

import numpy as np
from GP.genetic_algorithm import GeneticProgram

import os

from config import Config


def target_function(x):
    return x ** 3 + x ** 2 + x
    # return sin(x ** 2)


def generate_dataset():
    x_values = np.linspace(-1, 1, 21)  # evenly spaced points between -1 and 1
    y_values = []
    for x in range(21):
        y_values.append(target_function(x_values[x]))
    dataset = list(zip(x_values, y_values))

    return dataset


def add(x, y): return x + y


def subtract(x, y): return x - y


def multiply(x, y): return x * y


def divide(x, y): return x / y if y != 0 else 1  # Handle division by zero


def sin(x):
    return math.sin(x)


def cos(x):
    return math.cos(x)


def exp(x):
    # safe exponent functionality to avoid math range error
    if x > 10:
        x = 10
    elif x < -10:
        x = -10
    return math.exp(x)


def log(x):
    return math.log(x) if x > 0 else 1  # Handle log(0) or negative inputs gracefully


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    config = Config(config_path)

    # Access configuration variables
    independent_runs = config.get("independent_runs", 30)
    max_generations = config.get("max_generations", 200)
    initial_depth = config.get("initial_depth", 5)
    final_depth = config.get("final_depth", 7)
    mutation_rate = config.get("mutation_rate", 0.1)
    elitism_size = config.get("elitism_size", 1)
    output_dir = config.get("output_dir", "results/")
    population_size = config.get("population_size", 126)
    project_name = config.get("project_name", "SGP_project")
    seed = config.get("seed", 42)
    use_semantics = config.get("use_semantics", True)
    adaptive_threshold = config.get("adaptive_threshold", True)
    semantic_threshold = config.get("semantic_threshold", 0.01)
    tournament_size = config.get("tournament_size", 7)
    verbose = config.get("verbose", True)

    random.seed(seed)

    functions = [add, subtract, multiply, divide, sin, cos, exp, log]
    terminals = ['x', 1]
    if verbose:
        print("Loaded Configuration:")
        for key, value in config.items.items():
            print(f"{key}: {value}")
        input("CONFIGURED. PRESS ENTER")

    print(f"Project '{project_name}' initialized with population size {population_size}.")

    for i in range(2):

        gp = GeneticProgram(use_semantics=use_semantics, adaptive_threshold=adaptive_threshold, semantic_threshold=semantic_threshold,
                            population_size=population_size, elitism_size=elitism_size, initial_depth=initial_depth, final_depth=final_depth,
                            functions=functions, terminals=terminals, dataset=generate_dataset(),
                            tournament_size=tournament_size)

        gp.evolve(generations=max_generations, mutation_rate=mutation_rate)
