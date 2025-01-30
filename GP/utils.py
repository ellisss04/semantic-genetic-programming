from matplotlib import pyplot as plt


def get_depth_iterative(root):
    if not root:
        return 0

    stack = [(root, 1)]  # Each item is (node, current_depth)
    max_depth = 0

    while stack:
        node, depth = stack.pop()
        if node:
            max_depth = max(max_depth, depth)
            # Push children to stack with incremented depth
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))

    return max_depth


def plot_semantic_space(reduced_semantics, fitness_values):
    plt.scatter(reduced_semantics[:, 0], reduced_semantics[:, 1], c=fitness_values, cmap='viridis')
    plt.colorbar(label='Fitness')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.title('Semantic Space Visualisation')
    ax = plt.gca()
    ax.set_xlim([-4, 4])
    ax.set_ylim([-3, 3])
    plt.show()


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
    plt.ylabel("Fitness (logged)")
    plt.title("Avg Fitness Progression")
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
