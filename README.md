# Semantic-Aware Genetic Programming

Welcome to the repository for **Semantic-Aware Genetic Programming**! This project explores the role of semantic diversity in Genetic Programming (GP) through innovative selection mechanisms, including static and adaptive semantic thresholds. The goal is to enhance the evolutionary process by balancing exploration and exploitation, achieving robust solutions to symbolic regression problems.

## Features

- **Selection Algorithms**:
  - Traditional GP Selection
  - Static Semantic-Aware Selection (SiS)
  - Adaptive Semantic-Aware Selection (ASiS)

- **Operators**:
  - Arithmetic: `add`, `subtract`, `multiply`, `divide`
  - Trigonometric: `sin`, `cos`
  - Exponential: `exp`
  - Logarithmic: `log`

- **Customisable Parameters**:
  - Population size, generations, crossover and mutation rates.
  - Tournament size and threshold settings.

- **Diversity Analysis**:
  - Semantic and syntactic diversity tracking.
  - PCA-based visualisation of semantic space.

- **Symbolic Regression Problems**:
  - A suite of test problems with varying complexity, including polynomials and trigonometric functions.

## Repository Structure

```
|-- src/
|   |-- diversity.py        # Diversity tracking and calculations
|   |-- genetic_algorithm.py # Core GP implementation
|   |-- individual.py        # Individual representation
|   |-- node.py              # GP tree structure
|   |-- utils.py             # Helper functions
|-- experiments/
|   |-- datasets/            # Input datasets for symbolic regression
|   |-- results/             # Outputs and visualisations
|-- README.md                # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/semantic-aware-gp.git
   cd semantic-aware-gp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run experiments:
   ```bash
   python src/genetic_algorithm.py
   ```

## Usage

### Configuring Experiments
All experimental parameters are defined in `config.yaml`. Modify this file to set population size, number of generations, thresholds, and other parameters.

### Running the Algorithm
To execute a symbolic regression experiment:
```bash
python src/genetic_algorithm.py --config config.yaml
```

### Analysing Results
- Results are stored in the `experiments/results/` directory.
- Use provided scripts to generate PCA plots and diversity metrics:
  ```bash
  python src/utils.py --plot-pca
  ```

## Key Results
- **Adaptive Thresholds (ASiS):** Achieve higher hit rates and faster convergence for complex problems.
- **Semantic Diversity:** PCA visualisations highlight improved exploration with semantic-aware methods.

## Contributing
Contributions are welcome! Please submit issues or pull requests to enhance the project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or suggestions, feel free to reach out:
- **Email**: your.email@example.com
- **GitHub Issues**: [Submit an issue](https://github.com/yourusername/semantic-aware-gp/issues)
