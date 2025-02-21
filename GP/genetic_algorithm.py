import math
import os
import random
import copy
import time

from tqdm import tqdm
from math import sqrt
from typing import List, Callable, Any

from sklearn.decomposition import PCA

from GP.node import Node
from GP.individual import Individual
from GP.utils import *
from GP.diversity import set_fitness_diversity, set_semantic_diversity, track_unique_individuals


class GeneticProgram:
    """
    A class representing a Genetic Programming framework for evolving solutions to problems.
    This framework uses a population of tree-based programs, applying genetic operators like
    selection, crossover, and mutation to evolve solutions over generations.
    """

    def __init__(self,
                 config: str,
                 run_number: int,
                 output_dir: str,
                 max_generations: int,
                 use_semantics: bool,
                 adaptive_threshold: bool,
                 semantic_threshold: float,
                 population_size: int,
                 tournament_size: int,
                 elitism_size: int,
                 mutation_rate: float,
                 crossover_rate: int,
                 initial_depth: int,
                 final_depth: int,
                 max_final_depth: int,
                 functions: List[Callable],
                 terminals: List[Any],
                 dataset: List[tuple]):
        """
        Initialize the GeneticProgram instance.

        Args:
            population_size (int): Number of individuals in the population.
            functions (List[Callable]): List of functions (e.g., add, subtract) used as operators.
            terminals (List[Any]): List of terminals (e.g., variables and constants) used as operands.
        """
        self.config_path = config
        self.run_number = run_number
        self.output_dir = output_dir
        self.max_generations = max_generations
        self.use_semantics = use_semantics
        self.adaptive_threshold = adaptive_threshold
        self.max_semantic_threshold = semantic_threshold
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.initial_depth = initial_depth
        self.final_depth = final_depth
        self.max_final_depth = max_final_depth
        self.functions = functions
        self.terminals = terminals
        self.dataset = dataset

        self.start_time = None
        self.end_time = None
        self.generation = None
        self.function_arity_map = {}
        self.population = self.ramped_half_and_half()
        self.min_semantic_threshold = 0.01
        self.current_threshold = None
        self.hit_threshold = 0.05

        # Class arrays
        self.evaluated_nodes = []
        self.best_fitness_values = []
        self.avg_fitness_values = []
        self.min_fitness_values = []
        self.max_fitness_values = []
        self.median_fitness_values = []
        self.semantic_diversity_values = []
        self.fitness_diversity_values = []

        self.set_function_map()

    def set_function_map(self):
        for func in self.functions:
            self.function_arity_map.update({func: func.__code__.co_argcount})

    def get_best_individual(self):
        best_fitness = -math.inf
        best_individual = None
        for ind in self.population:
            if ind.fitness > best_fitness:
                best_individual = ind
                best_fitness = ind.fitness
        return best_individual

    def get_pop_semantic_vectors(self):
        semantic_vectors = []
        for ind in self.population:
            semantic_vectors.append(ind.semantic_vector)
        return semantic_vectors

    def get_pop_fitness_values(self):
        fitness_values = []
        for ind in self.population:
            fitness_values.append(ind.fitness)
        return fitness_values

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
        Ensures the population size matches self.population_size exactly.
        """
        population = []
        depths = list(range(self.initial_depth, self.final_depth + 1))
        num_depths = len(depths)
        individuals_per_group = self.population_size // (2 * num_depths)
        total_individuals = 2 * individuals_per_group * num_depths

        # Divide population into full and grow methods
        for depth in depths:
            for _ in range(individuals_per_group):
                # Full method
                population.append(Individual(self.generate_random_tree(depth, depth, method="full")))
                # Grow method
                population.append(Individual(self.generate_random_tree(depth, depth, method="grow")))

        # Add remaining individuals to match self.population_size
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            method = random.choice(["full", "grow"])
            depth = random.choice(depths)
            population.append(Individual(self.generate_random_tree(depth, depth, method=method)))

        random.shuffle(population)
        return population

    def select_parents(self):
        """
        Select two parents based on the selection method in use.
        """
        parent1 = self.select()
        if self.use_semantics:
            parent2 = self.semantic_selection(parent1)
        else:
            parent2 = self.select()
        if random.random() > 0.5:
            return parent1, parent2
        else:
            return parent2, parent1

    def select(self) -> Individual:
        """
        Select an individual from the population using tournament selection.

        Returns:
            Individual: The selected individual.
        """
        population_copy = list(self.population)
        tournament = random.sample(population_copy, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def semantic_selection(self, parent_1: Individual) -> Individual:

        best_candidate = random.choice(self.population)
        best_fitness = best_candidate.fitness

        for _ in range(self.tournament_size):
            competitor = random.choice(self.population)
            # if semantically different
            if parent_1.semantic_vector == competitor.semantic_vector:
                break
            if self.check_semantic_difference(parent_1.semantic_vector, competitor.semantic_vector):
                if competitor.fitness > best_fitness:
                    best_fitness = competitor.fitness
                    best_candidate = competitor

        return best_candidate

    # def semantic_selection(self, parent_1: Individual) -> Individual:
    #     parent_semantics = parent_1.semantic_vector
    #     # Filter semantically diverse individuals
    #     diverse_candidates = [
    #         ind for ind in self.population if self.check_semantic_difference(parent_semantics, ind.semantic_vector)
    #     ]
    #
    #     if diverse_candidates:
    #         # Perform tournament selection among semantically diverse individuals
    #         competitors = random.sample(diverse_candidates, min(self.tournament_size, len(diverse_candidates)))
    #         best_candidate = max(competitors, key=lambda ind: ind.fitness)
    #     else:
    #         # Fallback: Use standard tournament selection
    #         competitors = random.sample(self.population, self.tournament_size)
    #         best_candidate = max(competitors, key=lambda ind: ind.fitness)
    #
    #     return best_candidate

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

    def linear_decay(self):
        decay_rate = (self.min_semantic_threshold - self.max_semantic_threshold) / self.max_generations

        return self.max_semantic_threshold + decay_rate * self.generation

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

        parent1_crossover_depth = random.randint(1, get_tree_depth(parent1_copy.tree))

        parent2_crossover_depth = random.randint(1, get_tree_depth(parent2_copy.tree))

        offspring = self.subtree_crossover(parent1_copy.tree, parent2_copy.tree,
                                           parent1_crossover_depth, parent2_crossover_depth)

        return Individual(offspring)

    @staticmethod
    def subtree_crossover(parent1, parent2, parent1_crossover_depth, parent2_crossover_depth):
        # Get all nodes at the specified depth
        nodes1 = parent1.get_nodes_at_depth(parent1_crossover_depth)
        nodes2 = parent2.get_nodes_at_depth(parent2_crossover_depth)

        subtree1 = random.choice(nodes1)
        subtree2 = random.choice(nodes2)

        # avoids circular dependencies
        subtree2.assign_new_uuids(subtree2)

        offspring = parent1.replace_subtree(subtree1, subtree2)

        return offspring

    def apply_crossover(self, parent1, parent2):
        """
        Apply crossover between two parents or clone one of them.
        """
        if random.random() < self.crossover_rate:
            # Randomize crossover direction
            if random.random() > 0.5:
                return self.crossover(parent1, parent2)
            else:
                return self.crossover(parent2, parent1)
        # Clone one of the parents
        return parent1.clone() if random.random() < 0.5 else parent2.clone()

    def mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual by replacing nodes randomly; Using point mutation

        Args:
            individual (Individual): Individual to mutate.

        Returns:
            Individual: Mutated individual.
        """

        def mutate_node(node: Node, depth: int) -> Node:
            if random.random() < self.mutation_rate:
                if callable(node.value):  # Replace operator
                    arity = len(node.children)
                    # Filter functions by matching arity
                    compatible_functions = [func for func in self.functions if self.function_arity_map[func] == arity]
                    new_func = random.choice(compatible_functions)
                    new_children = [mutate_node(child, child.get_depth() - 1) for child in node.children]
                    mutated_node = Node(new_func, new_children)
                    mutated_node.update_children()
                    return Node(new_func, [mutate_node(child, child.get_depth() - 1)
                                           for child in node.children])
                else:  # Replace operand
                    new_terminal = random.choice(self.terminals)
                    while node.value == new_terminal:
                        new_terminal = random.choice(self.terminals)
                    return Node(new_terminal)

            if callable(node.value):  # Recurse on children
                node.children = [mutate_node(child, child.get_depth() - 1) for child in node.children]
                node.update_children()
            return node

        mutated_tree = mutate_node(individual.tree, individual.get_depth())
        return Individual(mutated_tree)

    def elitism(self) -> List[Individual]:
        # Sort the population by fitness in descending order
        sorted_population = sorted(self.population, key=lambda candidate: candidate.fitness, reverse=True)
        # Return the top x candidates
        elites = sorted_population[:self.elitism_size]
        for ind in self.population:
            if ind in elites:
                self.population.remove(ind)
        return elites

    def fitness_function_rmse(self, ind: Individual):
        total_error = 0
        total_node_count = 0

        # Reset the semantic vector before evaluation

        for x_value, y_value in self.dataset:
            variables = {'x': x_value}  # Map 'x' to the current x_value in the dataset
            y_pred, node_count = ind.evaluate(variables)  # Evaluate the individual's tree (f(x))
            ind.set_semantics(y_pred)  # Store the prediction in the semantic vector
            error = (y_pred - y_value) ** 2
            total_error += error
            total_node_count += node_count

        mse = total_error / len(self.dataset)  # Calculate the mean squared error
        rmse = sqrt(mse)

        if len(ind.semantic_vector) != 21:
            raise ValueError

        return -rmse, total_node_count

    def set_fitness_if_none(self):
        for individual in self.population:
            if individual.fitness is None:  # Only evaluate if fitness has not been assigned
                fitness, node_count = self.fitness_function_rmse(individual)
                individual.set_fitness(fitness)

    def genetic_operations(self):
        total_node_count = 0
        new_individuals = []

        for _ in range(2):
            parent1, parent2 = self.select_parents()

            child = self.apply_crossover(parent1, parent2)

            if child.get_depth() > self.max_final_depth:
                child.tree.prune(self.terminals, self.max_final_depth)

            child = self.mutate(child)

            if child.get_depth() > self.max_final_depth:
                raise ValueError

            # Evaluate fitness and update node count
            fitness, node_count = self.fitness_function_rmse(child)
            child.set_fitness(fitness)
            total_node_count += node_count

            new_individuals.append(child)

            if child.get_depth() > self.max_final_depth:
                print(child.get_depth())
                print(f'Depth over for child {child}')
        return new_individuals, total_node_count

    def steady_state_population(self):

        self.current_threshold = self.sigmoid_decay()
        # self.current_threshold = self.linear_decay()

        elites = self.elitism()

        new_individuals, total_node_count = self.genetic_operations()

        self.evaluated_nodes.append(total_node_count)
        self.population += elites

        sorted_population = sorted(self.population, key=lambda candidate: candidate.fitness, reverse=False)
        worst = sorted_population[:2]
        for ind in sorted_population:
            if ind in worst:
                self.population.remove(ind)
        self.population.append(new_individuals[0])
        self.population.append(new_individuals[1])

        random.shuffle(self.population)

    def evolve(self):
        """
        Run the genetic programming evolutionary process.
        """
        tqdm_loop = tqdm(range(self.max_generations), desc="Evolving", unit="Gen")
        self.start_time = time.time()
        best_fitness = None
        for generation in tqdm_loop:
            self.generation = generation
            self.set_fitness_if_none()

            metrics = self.get_fitness_metrics()

            best_fitness = metrics['best_fitness']
            mean_fitness = metrics['avg_fitness']

            # self.semantic_diversity_values.append(set_semantic_diversity(self.population))
            # self.fitness_diversity_values.append(set_fitness_diversity(self.population))

            if self.generation == self.max_generations/2 or self.generation == 1:
                fitness_values = self.get_pop_fitness_values()
                semantic_vectors = self.get_pop_semantic_vectors()

                # plot_semantic_space(semantic_vectors, fitness_values)
            tqdm_loop.set_description(f"Evolving Run {self.run_number + 1} - Best Fitness = {best_fitness} - "
                                      f"Mean Fitness {mean_fitness}.")

            self.steady_state_population()

        self.end_time = time.time()

        # self.post_processing()
        self.write_receipt()

        if abs(best_fitness) < self.hit_threshold:
            return 1
        return 0

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

    def post_processing(self):
        semantic_vectors = self.get_pop_semantic_vectors()
        fitness_values = self.get_pop_fitness_values()
        avg_fitness_values_abs = []

        for fitness in self.avg_fitness_values:
            avg_fitness_values_abs.append(abs(fitness))

        plot_semantic_space(semantic_vectors,
                            fitness_values)

        plot_semantic_heatmap(self.population)

        plot_fitness(self.max_generations,
                     self.median_fitness_values,
                     title="Median fitness across generations",
                     y_axis="Median fitness value")

        plot_fitness(self.max_generations,
                     avg_fitness_values_abs,
                     title="Mean average fitness across generations (logged)",
                     y_axis="Mean fitness value",
                     y_scale="log")

        # plot_semantic_diversity(self.max_generations,
        #                         self.semantic_diversity_values)
        #
        # plot_fitness_diversity(self.max_generations,
        #                        self.fitness_diversity_values)

    def write_receipt(self):
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        metrics = self.get_fitness_metrics()
        final_median_fitness = metrics['median_fitness']
        final_mean_fitness = metrics['avg_fitness']
        best_individual = self.get_best_individual()
        unique_individuals = track_unique_individuals(self.population)

        receipt_content = [
            f"RUN {self.run_number + 1}",
            "=" * 30,
            "Experiment Metrics:",
            f"Final Median Fitness: {final_median_fitness}",
            f"Final Mean Fitness: {final_mean_fitness}",
            f"Best Individual Tree: {best_individual}",
            f"Best Individual Fitness: {best_individual.fitness}",
            f"Ratio of Unique Individuals : Population: {unique_individuals}",
            f"Time to Complete: {self.end_time - self.start_time}s",
            "\n"
        ]

        receipt_filename = os.path.join(self.output_dir, "experiment_receipt.txt")

        try:
            with open(receipt_filename, 'a') as receipt_file:
                receipt_file.write("\n")
                receipt_file.write("\n".join(receipt_content))
        except IOError as e:
            print(f"Error writing to receipt file: {e}")
