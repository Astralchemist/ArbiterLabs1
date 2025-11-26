import random
import copy
import numpy as np
from typing import Dict, List, Any, Optional, Union

class Individual:
    def __init__(self, config: Dict[str, Any], result: Optional[float] = None):
        self.config = config
        self.result = result

    def __str__(self):
        return f"config: {self.config}, result: {self.result}"

class EvolutionaryOptimizer:
    """
    Naive Evolution Optimizer.
    It randomly initializes a population based on the search space.
    For each generation, it chooses better ones and does some mutation.
    """

    def __init__(self, search_space: Dict[str, Dict[str, Any]], population_size: int = 32, optimize_mode: str = 'maximize', mutation_prob: float = 0.1):
        """
        Parameters
        ----------
        search_space: dict
            Defines the search space. Example:
            {
                'param1': {'type': 'uniform', 'low': 0.0, 'high': 1.0},
                'param2': {'type': 'choice', 'values': [1, 2, 3, 4]},
                'param3': {'type': 'randint', 'low': 1, 'high': 10}
            }
        population_size: int
            The size of the population.
        optimize_mode: str
            'maximize' or 'minimize'.
        mutation_prob: float
            Probability of mutating a parameter.
        """
        self.search_space = search_space
        self.population_size = population_size
        self.optimize_mode = optimize_mode
        self.mutation_prob = mutation_prob
        
        self.population: List[Individual] = []
        self.random_state = np.random.RandomState()
        
        # Initialize population
        for _ in range(self.population_size):
            self._random_generate_individual()

    def _random_value(self, spec: Dict[str, Any]):
        if spec['type'] == 'uniform':
            return self.random_state.uniform(spec['low'], spec['high'])
        elif spec['type'] == 'choice':
            return self.random_state.choice(spec['values'])
        elif spec['type'] == 'randint':
            return self.random_state.randint(spec['low'], spec['high'])
        elif spec['type'] == 'loguniform':
            return np.exp(self.random_state.uniform(np.log(spec['low']), np.log(spec['high'])))
        else:
            raise ValueError(f"Unknown parameter type: {spec['type']}")

    def _random_generate_individual(self):
        config = {}
        for key, spec in self.search_space.items():
            config[key] = self._random_value(spec)
        self.population.append(Individual(config=config))

    def suggest(self) -> Dict[str, Any]:
        """
        Suggest a new configuration to evaluate.
        """
        # If we have individuals without results, return one of them
        for ind in self.population:
            if ind.result is None:
                return ind.config

        # Evolution step
        # Randomly choose two individuals
        if len(self.population) < 2:
             # Should not happen if population_size >= 2
             return self.population[0].config
             
        candidates = self.random_state.choice(self.population, 2, replace=False)
        ind1, ind2 = candidates[0], candidates[1]
        
        # Determine better individual
        if self.optimize_mode == 'maximize':
            better = ind1 if ind1.result > ind2.result else ind2
            worse = ind2 if ind1.result > ind2.result else ind1
        else:
            better = ind1 if ind1.result < ind2.result else ind2
            worse = ind2 if ind1.result < ind2.result else ind1
            
        # Create offspring from better individual
        new_config = copy.deepcopy(better.config)
        
        # Mutate
        for key, spec in self.search_space.items():
            if self.random_state.rand() < self.mutation_prob:
                new_config[key] = self._random_value(spec)
                
        # Replace worse individual with new offspring
        # We need to find the index of the worse individual in the population to replace it
        # But wait, we should strictly follow the logic: 
        # "The worst of the pair will be removed. Copy the best of the pair and mutate it to generate a new individual."
        
        # Remove worse
        self.population.remove(worse)
        
        # Add new individual
        new_ind = Individual(config=new_config)
        self.population.append(new_ind)
        
        return new_config

    def report_result(self, config: Dict[str, Any], result: float):
        """
        Report the result of a configuration.
        """
        # Find the individual with this config and update result
        found = False
        for ind in self.population:
            if ind.config == config:
                ind.result = result
                found = True
                break
        
        if not found:
            # This might happen if we suggest a config, but before reporting, 
            # the individual was removed in another suggest call (if concurrent).
            # But here we assume sequential usage for simplicity or that suggest returns the config of the individual it just added.
            # In our suggest implementation, we add the new individual to population before returning.
            # However, if we are just filling initial population, we return existing configs.
            pass

    def get_best_config(self) -> Dict[str, Any]:
        valid_pop = [ind for ind in self.population if ind.result is not None]
        if not valid_pop:
            return None
            
        if self.optimize_mode == 'maximize':
            best_ind = max(valid_pop, key=lambda x: x.result)
        else:
            best_ind = min(valid_pop, key=lambda x: x.result)
            
        return best_ind.config

