import unittest
from GP import GeneticProgram, Individual, Node


# Define the functions and terminals for testing
def add(x, y): return x + y


def subtract(x, y): return x - y


def multiply(x, y): return x * y


def divide(x, y): return x / y if y != 0 else 1  # Safe divide


class TestGeneticProgram(unittest.TestCase):
    #
    @classmethod
    def setUpClass(cls):
        functions = [add, subtract, multiply, divide]
        terminals = ['x', 'y', 1, 2, 3]
        cls.gp = GeneticProgram(population_size=10, max_depth=3, functions=functions, terminals=terminals)

    def test_tree_generation(self):
        """Test that a tree has a correct structure with functions having the right number of children."""
        tree = self.gp.generate_random_tree(depth=3)
        self.assertIsInstance(tree, Node)

        def check_node(node):
            if callable(node.value):  # If it's a function node
                self.assertEqual(len(node.children), node.value.__code__.co_argcount)
                for child in node.children:
                    check_node(child)

        check_node(tree)  # Recursive structure check

    def test_individual_evaluation(self):
        """Test that an individual can be evaluated correctly."""
        tree = Node(add, [Node('x'), Node('y')])
        individual = Individual(tree)
        result = individual.evaluate({'x': 2, 'y': 3})
        self.assertEqual(result, 5)  # Expect 2 + 3 = 5

    def test_fitness_function(self):
        """Test the fitness function with a known output."""
        tree = Node(add, [Node(1), Node(1)])
        individual = Individual(tree)
        fitness = self.gp.fitness_function(individual)
        self.assertIsInstance(fitness, float)  # Fitness should be a float

    def test_crossover(self):
        """Test that crossover produces a valid individual."""
        parent1 = Individual(self.gp.generate_random_tree(depth=3))
        parent2 = Individual(self.gp.generate_random_tree(depth=3))
        child = self.gp.crossover(parent1, parent2)
        self.assertIsInstance(child, Individual)
        self.assertIsInstance(child.tree, Node)

    def test_mutation(self):
        """Test that mutation produces a valid individual."""
        individual = Individual(self.gp.generate_random_tree(depth=3))
        mutated = self.gp.mutate(individual, mutation_rate=1.0)  # Force mutation
        self.assertIsInstance(mutated, Individual)
        self.assertIsInstance(mutated.tree, Node)
        self.assertNotEqual(str(mutated.tree), str(individual.tree))  # Mutation likely changes the tree

    def test_evolution(self):
        """Test that evolution progresses through generations without error."""
        # We expect each generation to print fitness information without errors
        self.gp.evolve(generations=5, mutation_rate=0.1)
        for individual in self.gp.population:
            self.assertIsNotNone(individual.fitness)
            self.assertIsInstance(individual.fitness, float)


if __name__ == "__main__":
    unittest.main()
