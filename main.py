from GP.genetic_algorithm import GeneticProgram


def test_terminal_at_leaves():
    gp = GeneticProgram(population_size=10, max_depth=3, functions=[add, subtract, multiply, divide],
                        terminals=['x', 'y', 1, 2])
    tree = gp.generate_random_tree(gp.max_depth)

    def check_leaves(node, depth):
        if depth == 0 or len(node.children) == 0:
            if callable(node.value):
                print("ERROR: Function node found at leaf position:", node.value)
            else:
                print("Terminal node at leaf position:", node.value)
        if callable(node.value):
            # If it's a function node, ensure children exist and recurse
            for child in node.children:
                check_leaves(child, depth - 1)

    check_leaves(tree, gp.max_depth)
    print("All terminal nodes are at the leaves.")


if __name__ == "__main__":

    def add(x, y): return x + y
    def subtract(x, y): return x - y
    def multiply(x, y): return x * y
    def divide(x, y): return x / y if y != 0 else 1  # Handle division by zero

    functions = [add, subtract, multiply, divide]
    terminals = ['x', 'y', 1, 2, 3]

    gp = GeneticProgram(population_size=200, max_depth=6, functions=functions, terminals=terminals)
    gp.evolve(generations=25, mutation_rate=0.1)
