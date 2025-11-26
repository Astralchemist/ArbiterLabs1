import numpy as np
import heapq
from typing import Dict, List, Callable, Tuple

class ParameterOptimizer:
    def __init__(self, evaluate_func: Callable, constraints: Dict = None,
                 generation_count: int = 5, population: int = 10,
                 top_n: int = 3, base_mutation_std: float = 0.1,
                 relative_std: bool = True, crossover_rate: float = 0.6,
                 seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.evaluate = evaluate_func
        self.constraints = constraints or {}
        self.generation_count = generation_count
        self.population = population
        self.top_n = top_n
        self.base_mutation_std = base_mutation_std
        self.relative_std = relative_std
        self.crossover_rate = crossover_rate
    
    def check_constraints(self, parameter_name: str, parameter_value: float) -> bool:
        if parameter_name not in self.constraints:
            return True
        for constraint in self.constraints[parameter_name]:
            if not constraint(parameter_value):
                return False
        return True
    
    def mutate_parameters(self, parameters: Dict) -> Dict:
        mutated_parameters = parameters.copy()
        for param_name, param_value in mutated_parameters.items():
            if self.relative_std:
                mutation_std = self.base_mutation_std * abs(param_value)
            else:
                mutation_std = self.base_mutation_std
            
            mutated_parameter = np.random.normal(loc=param_value, scale=mutation_std)
            
            if type(param_value) is int:
                mutated_parameter = int(np.round(mutated_parameter))
            
            if self.check_constraints(param_name, mutated_parameter):
                mutated_parameters[param_name] = mutated_parameter
        
        return mutated_parameters
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        if np.random.random() < self.crossover_rate:
            crossover_point = np.random.randint(0, len(parent1))
            offspring1 = dict(list(parent1.items())[:crossover_point] + 
                            list(parent2.items())[crossover_point:])
            offspring2 = dict(list(parent2.items())[:crossover_point] + 
                            list(parent1.items())[crossover_point:])
            return offspring1, offspring2
        else:
            return parent1, parent2
    
    def init_parameter_sets(self, base_parameter_set: Dict) -> List[Dict]:
        parameter_sets = [base_parameter_set]
        for _ in range(self.population - 1):
            parameter_sets.append(self.mutate_parameters(base_parameter_set))
        return parameter_sets
    
    def multiply(self, top_parameter_sets: List[Dict]) -> List[Dict]:
        crossover_parameter_sets = top_parameter_sets.copy()
        
        i = 0
        j = 1
        
        while len(crossover_parameter_sets) < self.population - 1:
            parent1 = top_parameter_sets[i % self.top_n]
            parent2 = top_parameter_sets[(i + j) % self.top_n]
            offspring1, offspring2 = self.crossover(parent1, parent2)
            crossover_parameter_sets.append(offspring1)
            crossover_parameter_sets.append(offspring2)
            i += 1
            if i == self.top_n // 2:
                j += 1
                i = 0
        
        if len(crossover_parameter_sets) < self.population:
            parent1 = top_parameter_sets[i % self.top_n]
            parent2 = top_parameter_sets[(i + j) % self.top_n]
            offspring1, _ = self.crossover(parent1, parent2)
            crossover_parameter_sets.append(offspring1)
        
        mutated_parameter_sets = [top_parameter_sets[0]]
        
        for i in range(1, self.population):
            mutated_parameter_sets.append(
                self.mutate_parameters(crossover_parameter_sets[i]))
        
        return mutated_parameter_sets
    
    def evaluate_parameter_sets(self, parameter_sets: List[Dict]) -> List[Tuple[float, Dict]]:
        evaluated_parameters = []
        for params in parameter_sets:
            score = self.evaluate(params)
            evaluated_parameters.append((score, params))
        return evaluated_parameters
    
    def selection(self, parameter_sets: List[Tuple[float, Dict]]) -> List[Tuple[float, Dict]]:
        return heapq.nlargest(self.top_n, parameter_sets, key=lambda x: x[0])
    
    def optimize_parameters(self, base_parameter_set: Dict) -> Tuple[Dict, float]:
        parameter_sets = self.init_parameter_sets(base_parameter_set)
        
        for generation in range(self.generation_count):
            top_parameter_sets_with_scores = self.selection(
                self.evaluate_parameter_sets(parameter_sets))
            
            top_parameter_sets = [params for _, params in top_parameter_sets_with_scores]
            parameter_sets = self.multiply(top_parameter_sets)
            
            print(f"Generation {generation + 1}: Best score = {top_parameter_sets_with_scores[0][0]:.4f}")
        
        return top_parameter_sets_with_scores[0][1], top_parameter_sets_with_scores[0][0]
