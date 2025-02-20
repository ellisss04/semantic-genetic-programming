import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def get_tree_depth(root):
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


def plot_semantic_space(semantic_vectors, fitness_values):
    pca = PCA(n_components=2)
    reduced_semantics = pca.fit_transform(semantic_vectors)
    plt.scatter(reduced_semantics[:, 0], reduced_semantics[:, 1], c=fitness_values, cmap='viridis')
    plt.colorbar(label='Fitness')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.title('Semantic Space Visualisation')
    ax = plt.gca()
    plt.show()


def plot_fitness(generations, avg_fitness, title, y_axis, y_scale="linear"):
    """
    Plot the fitness values over generations.

    Args:
        generations: Number of generations
        avg_fitness : List of average fitness values for each generation.
        title:
        y_axis:
        y_scale:
    """
    generations = list(range(generations))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label="Average Fitness", color="blue")
    plt.xlabel("Generation")
    plt.ylabel(y_axis)
    plt.yscale(y_scale)
    plt.title(title)
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


def plot_semantic_diversity(generations, semantic_diversity):
    plt.figure(figsize=(10, 6))
    generations = list(range(generations))
    plt.plot(generations, semantic_diversity, label="Semantic Diversity", color="green")
    plt.xlabel("Generation")
    plt.ylabel("Semantic diversity (logged)")
    plt.title("Semantic diversity progression over generations")
    plt.legend()
    plt.grid()
    plt.show()


def plot_fitness_diversity(generations, fitness_diversity):
    plt.figure(figsize=(10, 6))
    generations = list(range(generations))
    plt.plot(generations, fitness_diversity, label="Fitness Diversity", color="green")
    plt.xlabel("Generation")
    plt.ylabel("Fitness diversity (standard deviation)")
    plt.yscale("log")
    plt.title("Fitness diversity progression over generations")
    plt.legend()
    plt.grid()
    plt.show()


def plot_semantic_heatmap(population):
    distance_matrix = get_pairwise_distance(population)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap="viridis", annot=False)
    plt.title("Semantic Diversity Heatmap (Final Generation)")
    plt.xlabel("Individual Index")
    plt.ylabel("Individual Index")
    plt.show()


def get_pairwise_distance(population):
    """
    Compute pairwise semantic distances for a population using normalized Manhattan distance.

    Args:
        population (list): A list of individuals, each with a semantic_vector attribute.

    Returns:
        np.ndarray: A distance matrix of shape (n, n).
    """
    # Extract semantic vectors from the population
    semantics = np.array([np.ravel(ind.semantic_vector) for ind in population])  # Shape (n, m)

    n = len(population)
    pairwise_matrix = np.zeros((n, n))

    # Compute pairwise Manhattan distances
    for i in range(n):
        for j in range(i + 1, n):
            # Normalized Manhattan distance
            distance = np.sum(np.abs(semantics[i] - semantics[j])) / len(semantics[i])
            pairwise_matrix[i, j] = distance
            pairwise_matrix[j, i] = distance

    return pairwise_matrix


def check_circular_references(node, visited=None, path=None):
    """
    Check for circular references in a tree structure and return the path.

    Args:
        node: The current node in the tree.
        visited: A set to keep track of visited node IDs.
        path: A list to keep track of the traversal path.

    Raises:
        ValueError: If a circular reference is detected.
    """
    if visited is None:
        visited = set()
    if path is None:
        path = []

    if node.id in visited:
        raise ValueError(f"Circular reference detected at node: {node}. Path: {' -> '.join(path + [str(node)])}")

    visited.add(node.id)
    path.append(str(node))

    for child in node.children:
        check_circular_references(child, visited, path)

    path.pop()
