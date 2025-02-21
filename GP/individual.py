from typing import Any

from GP import node


class Individual:
    def __init__(self, tree: node):
        self.tree = tree
        self.fitness = None  # Fitness will be assigned later
        self.semantic_vector = []

    def set_fitness(self, fitness: float):
        self.fitness = fitness

    def get_depth(self):
        return self.tree.get_depth()

    def reset_semantics(self):
        """ Clears the semantic vector to prepare for a new evaluation."""
        self.semantic_vector = []

    def set_semantics(self, output: float):
        """
        Appends a new output value to the semantic vector.

        Args:
            output (float): The predicted output value to store.
        """
        self.semantic_vector.append(output)

    def evaluate(self, variables: dict) -> Any:
        return self.tree.evaluate(variables)

    def clone(self) -> 'Individual':
        from copy import deepcopy
        return Individual(deepcopy(self.tree))

    def __str__(self):
        return str(self.tree)
