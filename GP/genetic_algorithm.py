import math
import random
import copy
from tqdm import tqdm
from math import sqrt
from typing import List, Callable, Any

import numpy as np
from sklearn.decomposition import PCA

from GP.node import Node
from GP.individual import Individual
from GP.utils import plot_fitness, plot_evaluated_nodes, plot_semantic_space, get_depth_iterative


class GeneticProgram:
    """
    A class representing a Genetic Programming framework for evolving solutions to problems.
    This framework uses a population of tree-based programs, applying genetic operators like
    selection, crossover, and mutation to evolve solutions over generations.
    """

    def __init__(self, use_semantics: bool, adaptive_threshold: bool, semantic_threshold: float, population_size: int,
                 elitism_size: int, initial_depth: int, final_depth: int, functions: List[Callable],
                 terminals: List[Any], dataset, tournament_size: int):
        """
        Initialize the GeneticProgram instance.

        Args:
            population_size (int): Number of individuals in the population.
            functions (List[Callable]): List of functions (e.g., add, subtract) used as operators.
            terminals (List[Any]): List of terminals (e.g., variables and constants) used as operands.
        """
        self.max_generations = None
        self.generation = None
        self.use_semantics = use_semantics
        self.adaptive_threshold = adaptive_threshold
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.initial_depth = initial_depth
        self.final_depth = final_depth
        self.functions = functions
        self.function_arity_map = {}
        self.terminals = terminals
        self.population = self.ramped_half_and_half()
        self.dataset = dataset
        self.max_semantic_threshold = semantic_threshold
        self.min_semantic_threshold = 0.01
        self.current_threshold = None
        self.evaluated_nodes = []
        self.best_fitness_values = []
        self.avg_fitness_values = []
        self.min_fitness_values = []
        self.max_fitness_values = []
        self.median_fitness_values = []

        self.set_function_map()

    def set_function_map(self):
        for func in self.functions:
            self.function_arity_map.update({func: func.__code__.co_argcount})

    def generate_random_tree(self, depth, chosen_depth, method):
        """
        Recursively generate a random tree using 'grow' or 'full' initialization.

        Args:
            depth (int): Remaining depth for the tree.
            chosen_depth (int): Minimum depth the tree must reach.
            method (str): Initialization method ('grow' or 'full').

        Returns:
            Node: Generated random tree.
        """
        # Base case: Terminal node if at max depth or depth == 1
        depth_threshold = abs(depth-chosen_depth)
        if depth == 1 or (method == "grow" and depth_threshold > 2 and random.random() > 0.5):
            return Node(random.choice(self.terminals))  # Terminal node

        # Otherwise, generate a function node
        func = random.choice(self.functions)
        arity = func.__code__.co_argcount  # Determine number of children (arity)
        children = [self.generate_random_tree(depth - 1, chosen_depth, method) for _ in range(arity)]
        return Node(func, children)

    def ramped_half_and_half(self):
        """
        Generate a population using ramped half-and-half initialization.
        """
        population = []
        depths = list(range(self.initial_depth, self.final_depth + 1))

        # Divide population into full and grow methods
        for depth in depths:
            chosen_depth = depth
            for _ in range(self.population_size // (2 * len(depths))):
                # Full method
                population.append(Individual(self.generate_random_tree(depth, chosen_depth, method="full")))
                # Grow method
                population.append(Individual(self.generate_random_tree(depth, chosen_depth, method="grow")))

        random.shuffle(population)

        return population

    def select(self) -> Individual:
        """
        Select an individual from the population using tournament selection.

        Returns:
            Individual: The selected individual.
        """
        population_copy = copy.deepcopy(self.population)
        tournament = random.sample(population_copy, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def semantic_selection(self, parent_1: Individual) -> Individual:
        parent_semantics = parent_1.semantic_vector
        temp_population = self.population.copy()

        best_candidate = random.choice(temp_population)
        temp_population.remove(best_candidate)

        best_fitness = best_candidate.fitness

        for _ in range(self.tournament_size):
            competitor = random.choice(temp_population)
            temp_population.remove(competitor)
            competitor_semantics = competitor.semantic_vector
            # if semantically different
            if self.check_semantic_difference(parent_semantics, competitor_semantics):
                if competitor.fitness > best_fitness:
                    best_fitness = competitor.fitness
                    best_candidate = competitor

        return best_candidate

    def sigmoid_decay(self, steepness=1.0):
        """
        Calculate the decayed semantic threshold using a sigmoid function.

        Args:
            steepness (float): Controls how steep the decay is. Default is 1.0.

        Returns:
            float: The decayed semantic threshold.
        """
        midpoint = self.max_generations / 2
        decay = 1 / (1 + math.exp(-steepness * (self.generation - midpoint)))
        return self.max_semantic_threshold + (self.min_semantic_threshold - self.max_semantic_threshold) * decay

    def check_semantic_difference(self, semantics_1, semantics_2) -> bool:
        """
        Method to check whether two individuals are semantically different.

        Two individuals are considered semantically different if all corresponding
        values in their vectors differ by more than or equal to the threshold.
        If the configuration uses the adaptive threshold method, then it will use the decayed threshold
        """
        if not self.adaptive_threshold:
            return all(abs(val1 - val2) >= self.min_semantic_threshold
                       for val1, val2 in zip(semantics_1, semantics_2))
        else:
            return all(abs(val1 - val2) >= self.current_threshold
                       for val1, val2 in zip(semantics_1, semantics_2))

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Perform subtree crossover between two parents to create an offspring.

        Args:
            parent1 (Individual): First parent.
            parent2 (Individual): Second parent.

        Returns:
            Individual: New individual created from crossover.
        """

        parent1_copy = copy.deepcopy(parent1)
        parent2_copy = copy.deepcopy(parent2)

        parent1_crossover_depth = random.randint(1, get_depth_iterative(parent1_copy.tree))

        parent2_crossover_depth = random.randint(1, get_depth_iterative(parent2_copy.tree))

        offspring = self.subtree_crossover(parent1_copy.tree, parent2_copy.tree,
                                           parent1_crossover_depth, parent2_crossover_depth)

        return Individual(offspring)

    @staticmethod
    def subtree_crossover(self, parent1, parent2, parent1_crossover_depth, parent2_crossover_depth):
        # Get all nodes at the specified depth
        nodes1 = parent1.get_nodes_at_depth(parent1_crossover_depth)
        nodes2 = parent2.get_nodes_at_depth(parent2_crossover_depth)

        subtree1 = random.choice(nodes1)
        subtree2 = random.choice(nodes2)

        offspring = parent1.replace_subtree(subtree1, subtree2)

        return offspring

    def mutate(self, individual: Individual, mutation_rate: float) -> Individual:
        """
        Mutate an individual by replacing nodes randomly, ensuring arity constraints are respected.

        Args:
            individual (Individual): Individual to mutate.
            mutation_rate (float): Probability of mutating a node.

        Returns:
            Individual: Mutated individual.
        """

        def mutate_node(node: Node, depth: int) -> Node:
            if random.random() < mutation_rate:
                if callable(node.value):  # Replace operator
                    arity = len(node.children)
                    # Filter functions by matching arity
                    compatible_functions = [func for func in self.functions if self.function_arity_map[func] == arity]
                    new_func = random.choice(compatible_functions)
                    new_children = [mutate_node(child, get_depth_iterative(child) - 1) for child in node.children]
                    mutated_node = Node(new_func, new_children)
                    mutated_node.update_children()
                    return Node(new_func, [mutate_node(child, get_depth_iterative(child) - 1)
                                           for child in node.children])
                else:  # Replace operand
                    new_terminal = random.choice(self.terminals)
                    return Node(new_terminal)

            if callable(node.value):  # Recurse on children
                node.children = [mutate_node(child, get_depth_iterative(child) - 1) for child in node.children]
                node.update_children()
            return node

        mutated_tree = mutate_node(individual.tree, get_depth_iterative(individual.tree))
        return Individual(mutated_tree)

    def fitness_function(self, ind: Individual):
        total_error = 0  # Initialize total error for the individual
        total_node_count = 0

        for x_value, y_value in self.dataset:
            variables = {'x': x_value}  # Map 'x' to the current x_value in the dataset
            y_pred, node_count = ind.evaluate(variables)  # Evaluate the individual's tree (f(x))
            ind.set_semantics(y_pred)
            error = (y_pred - y_value) ** 2
            total_error += error
            total_node_count += node_count

        mse = total_error / len(self.dataset)  # Calculate the mean squared error
        rmse = sqrt(mse)

        return -rmse, total_node_count

    def elitism(self):
        # Sort the population by fit`ness in descending order
        sorted_population = sorted(self.population, key=lambda candidate: candidate.fitness, reverse=True)
        # Return the top x candidates
        elites = sorted_population[:self.elitism_size]
        for ind in self.population:
            if ind in elites:
                self.population.remove(ind)
        return elites

    def steady_state_population(self, mutation_rate: float):
        total_node_count = 0
        new_individuals = []
        for i in range(2):
            if self.use_semantics:
                parent1 = self.select()
                parent2 = self.semantic_selection(parent1)
            else:
                parent1, parent2 = self.select(), self.select()

            if random.random() > 0.5:
                child = self.crossover(parent1, parent2)
            else:
                child = self.crossover(parent2, parent1)

            child = self.mutate(child, mutation_rate)

            # Evaluate the child and assign its fitness
            fitness, node_count = self.fitness_function(child)
            child.set_fitness(fitness)  # Set the calculated fitness
            total_node_count += node_count

            new_individuals.append(child)

        self.evaluated_nodes.append(total_node_count)

        sorted_population = sorted(self.population, key=lambda candidate: candidate.fitness, reverse=False)
        worst = sorted_population[:2]
        for ind in sorted_population:
            if ind in worst:
                self.population.remove(ind)
        self.population.append(new_individuals[0])
        self.population.append(new_individuals[1])

    def set_new_population(self, mutation_rate: float):
        total_node_count = 0
        new_population = self.elitism()
        while len(new_population) < self.population_size:

            if self.use_semantics:
                parent1 = self.select()
                parent2 = self.semantic_selection(parent1)
            else:
                parent1, parent2 = self.select(), self.select()

            if random.random() > 0.5:
                child = self.crossover(parent1, parent2)
            else:
                child = self.crossover(parent2, parent1)

            child = self.mutate(child, mutation_rate)

            # Evaluate the child and assign its fitness
            fitness, node_count = self.fitness_function(child)
            child.set_fitness(fitness)  # Set the calculated fitness
            total_node_count += node_count

            new_population.append(child)

        self.evaluated_nodes.append(total_node_count)

        return new_population

    def get_fitness_metrics(self):
        # Filter out individuals with None fitness values
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]

        if not fitness_values:  # Check if there are valid fitness values
            return None, None

        # Calculate total, best, and average fitness
        total_fitness = sum(fitness_values)
        best_individual = max(self.population, key=lambda ind: ind.fitness, default=None)
        avg_fitness = total_fitness / len(fitness_values)

        # Calculate additional metrics
        median_fitness = np.median(fitness_values)
        min_fitness = np.min(fitness_values)
        max_fitness = np.max(fitness_values)

        # Store fitness values for this generation (for plotting)
        if best_individual is not None:
            self.best_fitness_values.append(best_individual.fitness)
        self.avg_fitness_values.append(avg_fitness)
        self.median_fitness_values.append(median_fitness)
        self.min_fitness_values.append(min_fitness)
        self.max_fitness_values.append(max_fitness)

        # Return a dictionary of fitness metrics
        metrics = {
            'best_fitness': best_individual.fitness if best_individual else None,
            'avg_fitness': avg_fitness,
            'median_fitness': median_fitness,
            'min_fitness': min_fitness,
            'max_fitness': max_fitness,
        }

        return metrics

    def evolve(self, generations: int, mutation_rate: float):
        """
        Run the genetic programming evolutionary process.

        Args:
            generations (int): Number of generations to evolve.
            mutation_rate (float): Probability of mutating nodes.
        """
        self.max_generations = generations
        tqdm_loop = tqdm(range(self.max_generations), desc="Evolving", unit="Gen")
        for generation in tqdm_loop:
            self.generation = generation
            # Evaluate fitness for all individuals
            for individual in self.population:
                if individual.fitness is None:  # Only evaluate if fitness has not been assigned
                    fitness, node_count = self.fitness_function(individual)
                    individual.set_fitness(fitness)

            metrics = self.get_fitness_metrics()

            best_fitness = metrics['best_fitness']
            median_fitness = metrics['median_fitness']
            mean_fitness = metrics['avg_fitness']

            tqdm_loop.set_description(f"Evolving - Best Fitness = {best_fitness} - "
                                      f"Median Fitness = {median_fitness}, Mean fitness {mean_fitness}.")

            self.current_threshold = self.sigmoid_decay()
            self.steady_state_population(mutation_rate)

        for i, val in enumerate(self.avg_fitness_values):
            self.avg_fitness_values[i] = abs(val)
        avg_fitness_log = np.log(self.avg_fitness_values)

        semantic_vectors = []
        fitness_values = []
        for ind in self.population:
            semantic_vectors.append(ind.semantic_vector)
            fitness_values.append(ind.fitness)

        pca = PCA(n_components=2)
        reduced_semantics = pca.fit_transform(semantic_vectors)

        plot_semantic_space(reduced_semantics, fitness_values)
        plot_fitness(self.max_generations, self.median_fitness_values)
        # plot_evaluated_nodes(self.evaluated_nodes)
