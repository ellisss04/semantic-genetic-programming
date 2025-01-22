from typing import Union, Callable, Optional, List
import uuid


class Node:
    def __init__(self, value: Union[str, Callable], children: Optional[List['Node']] = None):
        """
        Initialize a Node.

        Args:
            value (Union[str, Callable]): The value of the node, which can be a variable, constant, or function.
            children (Optional[List['Node']]): A list of child nodes.
        """
        self.id = uuid.uuid4()
        self.value = value
        self.children = children if children is not None else []

        # Set left and right only if children exist
        self.left = self.children[0] if len(self.children) > 0 else None
        self.right = self.children[1] if len(self.children) > 1 else None

    def validate_tree(self):
        """
        Recursively validate the entire tree to ensure that:
        - self.left matches self.children[0].
        - self.right matches self.children[1] (if applicable).

        Raises:
            ValueError: If any node in the tree has inconsistent children.
        """
        if self.left:
            if self.children[0] != self.left:
                raise ValueError(
                    f"Inconsistent state at node {self.value}: self.left ({self.left.value}) "
                    f"does not match self.children[0] ({self.children[0].value if self.children else None})."
                )

        if self.right:
            if self.children[1] != self.right:
                raise ValueError(
                    f"Inconsistent state at node {self.value}: self.right ({self.right.value}) "
                    f"does not match self.children[1] ({self.children[1].value if len(self.children) > 1 else None})."
                )

        # Recursively validate children
        for child in self.children:
            child.validate_tree()

    def evaluate(self, variables: dict) -> tuple:
        """
        Evaluate the node's value based on the provided variables.

        Args:
            variables (dict): A dictionary mapping variable names to their values.

        Returns:
            tuple: A tuple containing the evaluation result and the number of nodes evaluated.
        """
        if callable(self.value):
            # Determine the number of arguments (arity) of the function
            arity = self.value.__code__.co_argcount

            if len(self.children) != arity:
                raise ValueError(
                    f"Function {self.value.__name__} requires {arity} arguments, "
                    f"but got {len(self.children)} children."
                )

            # Evaluate the required number of children
            children_results, total_nodes = zip(*[child.evaluate(variables) for child in self.children[:arity]])
            return self.value(*children_results), sum(total_nodes) + 1

        # Terminal node case (variable or constant)
        return variables.get(self.value, self.value), 1

    def get_depth(self):
        current_depth = 0

        if self.left:
            current_depth = max(current_depth, self.left.get_depth())

        if self.right:
            current_depth = max(current_depth, self.right.get_depth())

        return current_depth + 1

    def get_nodes_at_depth(self, depth, current_depth=1):
        """
        Retrieve all nodes at the specified depth.

        Args:
            depth (int): Target depth to retrieve nodes.
            current_depth (int): Current depth in the tree during recursion.

        Returns:
            list: A list of nodes at the specified depth.
        """
        if current_depth == depth:
            return [self]  # Base case: Return current node if depth matches.

        nodes = []
        for child in self.children:
            nodes.extend(child.get_nodes_at_depth(depth, current_depth + 1))  # Collect nodes from children.
        return nodes

    def replace_subtree(self, target, replacement):
        if self.id == target.id:
            return replacement

        for i, child in enumerate(self.children):
            if child.id == target.id:
                self.update_attributes(replacement, i)
                return self
            else:
                self.children[i] = child.replace_subtree(target, replacement)
        return self

    def update_attributes(self, replacement, index):
        self.children[index] = replacement
        if index == 0:
            self.left = replacement
        elif index == 1:
            self.right = replacement
        else:
            raise IndexError


    def __str__(self):
        # TODO: string not representing tree properly
        if callable(self.value):
            return f"({self.value.__name__} {' '.join(map(str, self.children))})"
        return str(self.value)
