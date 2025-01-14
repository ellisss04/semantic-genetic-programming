from typing import Union, Callable, Optional, List


class Node:
    def __init__(self, value: Union[str, Callable], children: Optional[List['Node']] = None):
        self.value = value
        self.children = children if children is not None else []

        self.left = None
        self.right = None
        if self.children:
            self.left = self.children[0]
            self.right = self.children[1]

    def evaluate(self, variables: dict) -> tuple:
        """
        Evaluate the node's value based on the provided variables.

        Args:
            variables (dict): A dictionary mapping variable names to their values.

        Returns:
            tuple: A tuple containing the evaluation result and the number of nodes evaluated.
        """
        if callable(self.value) and len(self.children) == 2:
            if len(self.children) != 2:
                raise ValueError(
                    f"Function {self.value.__name__} requires 2 arguments, "
                    f"but got {len(self.children)} children."
                )
            # Evaluate children and sum up node counts
            children_results, total_nodes = zip(*[child.evaluate(variables) for child in self.children])
            return self.value(*children_results), sum(total_nodes) + 1

        # Terminal node case
        return variables.get(self.value, self.value), 1

    def get_depth(self):
        current_depth = 0

        if self.left:
            current_depth = max(current_depth, self.left.get_depth())

        if self.right:
            current_depth = max(current_depth, self.right.get_depth())

        return current_depth + 1

    def get_nodes_at_depth(self, depth, current_depth=1):
        if current_depth == depth:
            return [self]
        nodes = []
        for child in self.children:
            nodes.append(child)
        return nodes

    def replace_subtree(self, target, replacement):
        if self == target:
            return replacement

        for i, child in enumerate(self.children):
            if child == target:
                self.children[i] = replacement
                return self
            else:
                self.children[i] = child.replace_subtree(target, replacement)
        return self

    def __str__(self):
        if callable(self.value):
            return f"({self.value.__name__} {' '.join(map(str, self.children))})"
        return str(self.value)
