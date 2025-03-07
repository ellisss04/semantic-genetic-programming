import math
import random

import numpy as np

from GP.genetic_algorithm import GeneticProgram

import os

from config import Config


def target_function(x):
    return sin(x) + sin(x + x ** 2)
    # return 2 * (sin(x) * cos(x))


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


def divide(x, y):
    if abs(y) < 1:  # Avoid very small denominators
        return x  # Treat as no-op
    return x / y


def sin(x):
    return math.sin(x)


def cos(x):
    return math.cos(x)


def exp(x):
    # safe exponent functionality to avoid extreme outliers
    if x > 2:
        x = 2
    elif x < -2:
        x = -2
    return math.exp(x)


def log(x):
    return math.log(x) if x > 0 else 0


def sqrt(x):
    return math.sqrt(x) if x >= 0 else 0


def write_config_details(output_dir, config, verbose):
    """
    Write configuration details to a file in the output directory.

    Args:
        output_dir (str): Path to the output directory.
        config (Config): Configuration object containing experiment details.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "experiment_receipt.txt")

    with open(config_file, "w") as file:
        file.write("Configuration Details\n")
        file.write("=" * 30 + "\n")
        for key, value in config.items.items():
            file.write(f"{key}: {value}\n")
            if verbose:
                print(f"{key}: {value}")
    input("CONFIGURED. PRESS ENTER")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    config = Config(config_path)

    # Access configuration variables
    independent_runs = config.get("independent_runs", 30)
    max_generations = config.get("max_generations", 200)
    initial_depth = config.get("initial_depth", 5)
    final_depth = config.get("final_depth", 7)
    max_final_depth = config.get("max_final_depth", 9)
    crossover_rate = config.get("crossover_rate", 1.0)
    mutation_rate = config.get("mutation_rate", 0.1)
    elitism_size = config.get("elitism_size", 1)
    output_dir = config.get("output_dir", "results/")
    population_size = config.get("population_size", 126)
    project_name = config.get("project_name", "SGP_project")
    seed = config.get("seed", 42)
    use_semantics = config.get("use_semantics", True)
    adaptive_threshold = config.get("adaptive_threshold", True)
    max_semantic_threshold = config.get("max_semantic_threshold", 0.05)
    min_semantic_threshold = config.get("min_semantic_threshold", 0.01)
    tournament_size = config.get("tournament_size", 7)
    verbose = config.get("verbose", True)
    number_of_runs = config.get("independent_runs", 30)

    random.seed(seed)

    functions = [add, subtract, multiply, divide, sin, cos, log]
    terminals = ['x', 1]

    write_config_details(output_dir, config, verbose)

    print("=" * 30)
    total_hits = 0
    solved_generations = 0
    solved_count = 0
    for run in range(number_of_runs):
        gp = GeneticProgram(
            config=config_path,
            run_number=run,
            output_dir=output_dir,
            max_generations=max_generations,
            use_semantics=use_semantics,
            adaptive_threshold=adaptive_threshold,
            max_semantic_threshold=max_semantic_threshold,
            min_semantic_threshold=min_semantic_threshold,
            population_size=population_size,
            elitism_size=elitism_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            initial_depth=initial_depth,
            final_depth=final_depth,
            max_final_depth=max_final_depth,
            functions=functions,
            terminals=terminals,
            dataset=generate_dataset(),
            tournament_size=tournament_size,
        )

        hits, solved_generation = gp.evolve()
        total_hits += hits
        if solved_generation:
            solved_generations += solved_generation
            solved_count += 1

    print(f'number of hits {total_hits}')
    if solved_count != 0:
        print(f'average generation of hit {solved_generations / solved_count}')
