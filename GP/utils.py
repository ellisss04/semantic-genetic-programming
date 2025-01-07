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
    # plt.plot(generations, best_fitness, label="Best Fitness", color="red")
    plt.plot(generations, avg_fitness, label="Average Fitness", color="blue")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progression Over Generations")
    plt.legend()
    plt.grid()
    plt.show()
