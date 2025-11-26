# Genetic Algorithm Evolved Trading Strategies

This module contains a sophisticated genetic algorithm framework for evolving trading strategies.

## Components

### Core Modules

- `individual.py` - Individual representation with genes and trading pairs
- `population.py` - Population management for genetic algorithm
- `operators.py` - Genetic operators (crossover, mutation, selection)
- `main.py` - Main genetic algorithm runner
- `backtest.py` - Backtesting and fitness evaluation
- `strategy.py` - Base strategy implementation

## Features

- Multi-parameter optimization using genetic algorithms
- Tournament selection for population evolution
- Adaptive mutation strategies (noise, reset, scale)
- Trading pair optimization
- Fitness-based evaluation
- Parallel processing support

## Usage

```python
from individual import Individual
from population import Population
from operators import crossover, mutate, select_tournament

# Create initial population
population = Population.create_random(
    size=50,
    parameters=params,
    trading_pairs=pairs,
    num_pairs=10
)

# Evolve population
for generation in range(num_generations):
    # Evaluate fitness
    for ind in population.individuals:
        ind.fitness = evaluate(ind)
    
    # Selection and reproduction
    offspring = [select_tournament(population.individuals, 5) for _ in range(50)]
    
    # Apply genetic operators
    for i in range(0, len(offspring), 2):
        offspring[i], offspring[i+1] = crossover(offspring[i], offspring[i+1])
    
    for ind in offspring:
        mutate(ind, mutation_rate=0.1)
```

## Parameters

The genetic algorithm supports optimization of:
- Integer parameters (e.g., periods, thresholds)
- Decimal parameters (e.g., multipliers, ratios)
- Categorical parameters (e.g., indicator types)
- Boolean parameters (e.g., feature toggles)

## Strategy Evolution

The framework evolves trading strategies by:
1. Generating random initial population
2. Evaluating fitness using backtesting
3. Selecting best performers via tournament selection
4. Creating offspring through crossover
5. Introducing variations via mutation
6. Repeating until convergence
