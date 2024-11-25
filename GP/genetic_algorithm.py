import random
from typing import List, Callable, Any

from GP.node import Node
from GP.individual import Individual


class GeneticProgram:
    def __init__(self, population_size: int, max_depth: int, functions: List[Callable], terminals: List[Any]):
        self.population_size = population_size
        self.max_depth = max_depth
        self.functions = functions
        self.terminals = terminals
        self.population = self.initialize_population()

    # Initialize random population
    def initialize_population(self) -> List[Individual]:
        return [Individual(self.generate_random_tree(self.max_depth)) for _ in range(self.population_size)]

    # Generate a random tree recursively
    def generate_random_tree(self, depth: int) -> Node:
        # Base case: If at maximum depth, select a terminal (operand)
        if depth == 0 or (depth > 1 and random.random() > 0.5):
            return Node(random.choice(self.terminals))  # Select an operand only

        # Recursive case: Select a function and create the appropriate number of child nodes
        func = random.choice(self.functions)  # Select an operator
        arg_count = func.__code__.co_argcount  # Get the number of arguments the function expects
        children = [self.generate_random_tree(depth - 1) for _ in range(arg_count)]
        return Node(func, children)

    @staticmethod
    def fitness_function(individual: Individual) -> float:
        # Example: Negative sum of evaluated output for a dummy dataset
        example_data = [{'x': 1, 'y': 2}, {'x': -1, 'y': -2}]
        error = 0
        for data_point in example_data:
            result = individual.evaluate(data_point)
            target = sum(data_point.values())  # Dummy target for illustration
            error += abs(target - result)
        return float(-error)

    def select(self) -> Individual:
        tournament = random.sample(self.population, 3)
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child_tree = self.subtree_crossover(parent1.tree, parent2.tree)
        return Individual(child_tree)

    def subtree_crossover(self, tree1: Node, tree2: Node) -> Node:
        """
        Perform a type-aware subtree crossover, ensuring operators are swapped
        with operators and operands with operands.
        """

        # Base case: If one node is an operand and the other is an operator, do not swap; just return the original
        if callable(tree1.value) != callable(tree2.value):
            return tree1 if random.random() > 0.5 else tree2

        if callable(tree1.value) and callable(tree2.value):  # Both are operators
            # Create a new operator node with children recursively crossed over
            new_tree = Node(tree1.value,
                            [self.subtree_crossover(c1, c2) for c1, c2 in zip(tree1.children, tree2.children)])
        else:  # Both are operands
            new_tree = Node(tree1.value if random.random() > 0.5 else tree2.value)

        return new_tree

    def mutate(self, individual: Individual, mutation_rate: float) -> Individual:
        """Mutate an individual by randomly replacing a node with a new operand or operator of the same type."""

        def mutate_node(node: Node, depth: int) -> Node:
            if random.random() < mutation_rate:
                if callable(node.value):  # It's an operator, so replace with another operator
                    new_func = random.choice(self.functions)
                    return Node(new_func, [mutate_node(child, depth - 1) for child in node.children])
                else:
                    new_terminal = random.choice(self.terminals)
                    return Node(new_terminal)

            if callable(node.value):
                node.children = [mutate_node(child, depth - 1) for child in node.children]
            return node

        mutated_tree = mutate_node(individual.tree, self.max_depth)
        return Individual(mutated_tree)

    def evolve(self, generations: int, mutation_rate: float):
        for generation in range(generations):
            for individual in self.population:
                if individual.fitness is None:
                    individual.fitness = self.fitness_function(individual)

            # Calculate best and average fitness
            total_fitness = sum(ind.fitness for ind in self.population if ind.fitness is not None)
            best_individual = max(
                (ind for ind in self.population if ind.fitness is not None),
                key=lambda ind: ind.fitness,
                default=None
            )
            avg_fitness = total_fitness / len(self.population) if self.population else 0

            if best_individual is not None:
                print(f"Generation {generation}: Best Fitness = {best_individual.fitness}, "
                      f"Average Fitness = {avg_fitness}")

            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select(), self.select()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate)
                child.fitness = self.fitness_function(child)
                new_population.append(child)

            self.population = new_population
