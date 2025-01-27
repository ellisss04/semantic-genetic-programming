from matplotlib import pyplot as plt


def plot_fitness(generations, avg_fitness):
    """
    Plot the fitness values over generations.

    Args:
        generations: Number of generations
        avg_fitness : List of average fitness values for each generation.
    """
    generations = list(range(generations))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label="Average Fitness", color="blue")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Median Fitness Progression")
    plt.legend()
    plt.grid()
    plt.show()


def plot_evaluated_nodes(evaluated_nodes):
    plt.figure(figsize=(10, 6))
    generations = list(range(len(evaluated_nodes)))
    plt.plot(generations, evaluated_nodes, label="Evaluated Nodes", color="red")
    plt.xlabel("Generation")
    plt.ylabel("Number of evaluated nodes")
    plt.title("Evaluated Node Progression Over Generations")
    plt.legend()
    plt.grid()
    plt.show()
