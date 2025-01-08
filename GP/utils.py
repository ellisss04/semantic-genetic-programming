from matplotlib import pyplot as plt


def plot_fitness(best_fitness, avg_fitness):
    """
    Plot the fitness values over generations.

    Args:
        best_fitness: List of best fitness values for each generation.
        avg_fitness : List of average fitness values for each generation.
    """
    plt.figure(figsize=(10, 6))
    generations = list(range(len(best_fitness)))
    plt.plot(generations, avg_fitness, label="Average Fitness", color="blue")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progression Over Generations")
    plt.legend()
    plt.grid()
    plt.show()


def plot_evaluated_nodes(evaluated_nodes):
    plt.figure(figsize=(10, 6))
    generations = list(range(len(evaluated_nodes)))
    plt.plot(generations, evaluated_nodes, label="Semantics in Selection", color="red")
    plt.xlabel("Generation")
    plt.ylabel("Numbeer of evaluated nodes")
    plt.title("Evaluated Node Progression Over Generations")
    plt.legend()
    plt.grid()
    plt.show()
