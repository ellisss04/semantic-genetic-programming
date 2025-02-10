from typing import Any

from GP import node


class Individual:
    def __init__(self, tree: node):
        self.tree = tree
        self.fitness = None  # Fitness will be assigned later
        self.semantic_vector = []
        self.hits = 0

    def set_fitness(self, fitness: float):
        self.fitness = fitness

    def get_depth(self):
        return self.tree.get_depth()

    def set_semantics(self, output):
        self.semantic_vector.append(output)

    def evaluate(self, variables: dict) -> Any:
        return self.tree.evaluate(variables)

    def clone(self) -> 'Individual':
        from copy import deepcopy
        return Individual(deepcopy(self.tree))

    def __str__(self):
        return str(self.tree)
