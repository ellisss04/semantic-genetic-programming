from typing import Union, Callable, Any, Optional, List


class Node:
    def __init__(self, value: Union[str, Callable], children: Optional[List['Node']] = None):
        self.value = value
        self.children = children if children is not None else []

    def evaluate(self, variables: dict) -> Any:
        if callable(self.value) and len(self.children) == 2:
            if len(self.children) != 2:
                raise ValueError(
                    f"Function {self.value.__name__} requires 2 arguments, "
                    f"but got {len(self.children)} children.")
            return self.value(*[child.evaluate(variables) for child in self.children])

        # Terminal node case
        return variables.get(self.value, self.value)

    def __str__(self):
        if callable(self.value):
            return f"({self.value.__name__} {' '.join(map(str, self.children))})"
        return str(self.value)
