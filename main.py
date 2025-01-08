import numpy as np
from GP.genetic_algorithm import GeneticProgram

POPULATION_SIZE = 126
MAX_DEPTH = 5
GENERATIONS = 200
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 7


def test_terminal_at_leaves():
    gp = GeneticProgram(population_size=10, max_depth=2, functions=[add, subtract, multiply, divide],
                        terminals=['x', 'y', 1, 2])
    tree = gp.generate_random_tree(gp.max_depth)

    def check_leaves(node, depth):
        if depth == 0 or len(node.children) == 0:
            if callable(node.value):
                print("ERROR: Function node found at leaf position:", node.value)
            else:
                print("Terminal node at leaf position:", node.value)
        if callable(node.value):
            # If it's a function node, ensure children exist and recurse
            for child in node.children:
                check_leaves(child, depth - 1)

    check_leaves(tree, gp.max_depth)
    print("All terminal nodes are at the leaves.")


def target_function(x):
    return x ** 3 + x ** 2 + x


def generate_dataset():
    x_values = np.linspace(-1, 1, 20)  # 20 evenly spaced points between -10 and 10
    y_values = target_function(x_values)
    dataset = list(zip(x_values, y_values))

    return dataset


def add(x, y): return x + y


def subtract(x, y): return x - y


def multiply(x, y): return x * y


def divide(x, y): return x / y if y != 0 else 1  # Handle division by zero


if __name__ == "__main__":
    functions = [add, subtract, multiply, divide]
    terminals = ['x', 1]

    semantic_choice = False
    choice = input("USE SEMANTICS IN SELECTION? Y/N").upper()
    if choice == 'Y':
        semantic_choice = True
    input(f"STARTING ALGORITHM FOR POPULATION SIZE {POPULATION_SIZE}")
    gp = GeneticProgram(use_semantics=semantic_choice, population_size=POPULATION_SIZE, max_depth=MAX_DEPTH,
                        functions=functions, terminals=terminals, dataset=generate_dataset(),
                        tournament_size=TOURNAMENT_SIZE)

    gp.evolve(generations=GENERATIONS, mutation_rate=MUTATION_RATE)
