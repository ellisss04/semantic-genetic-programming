import random
from math import sqrt
from typing import List, Callable, Any

import numpy as np

from GP.node import Node
from GP.individual import Individual
from GP.utils import plot_fitness, plot_evaluated_nodes


class GeneticProgram:
    """
    A class representing a Genetic Programming framework for evolving solutions to problems.
    This framework uses a population of tree-based programs, applying genetic operators like
    selection, crossover, and mutation to evolve solutions over generations.
    """

    def __init__(self, use_semantics: bool, population_size: int, max_depth: int, min_depth:int, functions: List[Callable],
                 terminals: List[Any], dataset, tournament_size: int):
        """
        Initialize the GeneticProgram instance.

        Args:
            population_size (int): Number of individuals in the population.
            max_depth (int): Maximum depth of the trees representing individuals.
            functions (List[Callable]): List of functions (e.g., add, subtract) used as operators.
            terminals (List[Any]): List of terminals (e.g., variables and constants) used as operands.
        """
        self.use_semantics = use_semantics
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.functions = functions
        self.terminals = terminals
        self.population = self.initialize_population()
        self.dataset = dataset
        self.semantic_threshold = 0.01

        self.evaluated_nodes = []
        self.best_fitness_values = []
        self.avg_fitness_values = []

    def initialize_population(self) -> List[Individual]:
        """
        Generate an initial population of individuals with random trees.

        Returns:
            List[Individual]: List of randomly generated individuals.

        """
        return [Individual(self.generate_random_tree(self.max_depth, self.min_depth)) for _ in
                range(self.population_size)]

    def generate_random_tree(self, depth: int, min_depth: int) -> Node:
        """
        Recursively generate a random tree structure.

        Args:
            depth (int): Remaining depth for the tree.
            min_depth (int): The minimum depth of a tree before terminal nodes are added

        Returns:
            Node: Root node of the generated tree.
        """
        if depth == 0 or (depth <= min_depth and random.random() > 0.5):
            return Node(random.choice(self.terminals))  # Terminal node
        func = random.choice(self.functions)
        arg_count = func.__code__.co_argcount  # Number of arguments for the function
        children = [self.generate_random_tree(depth - 1, min_depth) for _ in range(arg_count)]
        return Node(func, children)

    def select(self) -> Individual:
        """
        Select an individual from the population using tournament selection.

        Returns:
            Individual: The selected individual.
        """
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def semantic_selection(self, parent_1: Individual):
        parent_semantics = parent_1.semantic_vector

        best_candidate = random.choice(self.population)
        best_fitness = best_candidate.fitness

        for i in range(self.tournament_size):
            competitor = random.choice(self.population)
            competitor_semantics = competitor.semantic_vector
            # if semantically different
            if self.check_semantic_difference(parent_semantics, competitor_semantics):
                if competitor.fitness > best_fitness:
                    best_fitness = competitor.fitness
                    best_candidate = competitor

        return best_candidate

    def check_semantic_difference(self, semantics_1, semantics_2) -> bool:
        """
        Method to check whether two individuals are semantically different.

        Two individuals are considered semantically different if all corresponding
        values in their vectors differ by more than or equal to the threshold.
        """
        return all(abs(val1 - val2) >= self.semantic_threshold
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
        child_tree = self.subtree_crossover(parent1.tree, parent2.tree)

        return Individual(child_tree)

    def subtree_crossover(self, tree1: Node, tree2: Node) -> Node:
        """
        Perform subtree crossover between two nodes.

        Args:
            tree1 (Node): Subtree of the first parent.
            tree2 (Node): Subtree of the second parent.

        Returns:
            Node: New subtree generated from crossover.
        """
        if callable(tree1.value) != callable(tree2.value):  # Ensure type match
            return tree1 if random.random() > 0.5 else tree2
        if callable(tree1.value) and callable(tree2.value):  # Both are functions
            new_tree = Node(tree1.value,
                            [self.subtree_crossover(c1, c2) for c1, c2 in zip(tree1.children, tree2.children)])
        else:  # Both are terminals
            new_tree = Node(tree1.value if random.random() > 0.5 else tree2.value)
        return new_tree

    def mutate(self, individual: Individual, mutation_rate: float) -> Individual:
        """
        Mutate an individual by replacing nodes randomly.

        Args:
            individual (Individual): Individual to mutate.
            mutation_rate (float): Probability of mutating a node.

        Returns:
            Individual: Mutated individual.
        """

        def mutate_node(node: Node, depth: int) -> Node:
            if random.random() < mutation_rate:
                if callable(node.value):  # Replace operator
                    new_func = random.choice(self.functions)
                    return Node(new_func, [mutate_node(child, depth - 1) for child in node.children])
                else:  # Replace operand
                    new_terminal = random.choice(self.terminals)
                    return Node(new_terminal)
            if callable(node.value):  # Recurse on children
                node.children = [mutate_node(child, depth - 1) for child in node.children]
            return node

        mutated_tree = mutate_node(individual.tree, self.max_depth)
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

    def set_new_population(self, mutation_rate):
        new_population = []
        total_node_count = 0
        while len(new_population) < self.population_size:
            if self.use_semantics:
                parent1 = self.select()
                parent2 = self.semantic_selection(parent1)
            else:
                parent1, parent2 = self.select(), self.select()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, mutation_rate)

            # Evaluate the child and assign its fitness
            fitness, node_count = self.fitness_function(child)
            child.set_fitness(fitness)  # Set the calculated fitness
            total_node_count += node_count

            new_population.append(child)

        self.evaluated_nodes.append(total_node_count)

        return new_population

    def get_fitness_metrics(self):
        # Calculate total, best, and average fitness
        total_fitness = sum(ind.fitness for ind in self.population if ind.fitness is not None)
        best_individual = max(
            (ind for ind in self.population if ind.fitness is not None),
            key=lambda ind: ind.fitness,
            default=None
        )
        avg_fitness = total_fitness / len(self.population) if self.population else 0

        # Store fitness values for this generation
        if best_individual is not None:
            self.best_fitness_values.append(best_individual.fitness)
        self.avg_fitness_values.append(avg_fitness)

        return best_individual, avg_fitness

    def evolve(self, generations: int, mutation_rate: float):
        """
        Run the genetic programming evolutionary process.

        Args:
            generations (int): Number of generations to evolve.
            mutation_rate (float): Probability of mutating nodes.
        """
        for generation in range(generations):
            # Evaluate fitness for all individuals
            for individual in self.population:
                if individual.fitness is None:  # Only evaluate if fitness has not been assigned
                    fitness, node_count = self.fitness_function(individual)
                    individual.set_fitness(fitness)

            best_individual, avg_fitness = self.get_fitness_metrics()

            print(f"Generation {generation}: Best Fitness = {best_individual.fitness}, "
                  f"Average Fitness = {avg_fitness}")

            new_population = self.set_new_population(mutation_rate)
            self.population = new_population

        best_fitness_values = np.array(self.best_fitness_values)
        avg_fitness_values = np.array(self.avg_fitness_values)
        best_fitness_log = np.log(np.abs(best_fitness_values))
        avg_fitness_log = np.log(np.abs(avg_fitness_values))

        plot_fitness(best_fitness_log, avg_fitness_log)
        plot_evaluated_nodes(self.evaluated_nodes)
