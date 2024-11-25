from typing import Any

from GP import node


class Individual:
    def __init__(self, tree: node):
        self.tree = tree
        self.fitness = None  # Fitness will be assigned later

    def evaluate(self, variables: dict) -> Any:
        return self.tree.evaluate(variables)

    def __str__(self):
        return str(self.tree)
