import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Dict

class PortfolioGeneticAlgorithm:
    def __init__(self, population_size: int = 100, generations: int = 40,
                 crossover_prob: float = 0.5, mutation_prob: float = 0.2,
                 tournament_size: int = 3):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        
    def moving_average_crossover(self, data: pd.DataFrame, short_window: int, 
                                 long_window: int) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
        signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
        signals['signal'][short_window:] = np.where(
            signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        return signals
    
    def calculate_returns(self, data: pd.DataFrame, signals: pd.DataFrame, 
                         initial_capital: float = 10000) -> pd.DataFrame:
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions['stock'] = 100 * signals['signal']
        
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['holdings'] = positions['stock'] * data['Close']
        portfolio['cash'] = initial_capital - (positions.diff()['stock'] * data['Close']).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        return portfolio
    
    def evaluate_strategy(self, individual: List[int], data: pd.DataFrame) -> float:
        short_window = individual[0]
        long_window = individual[1]
        
        if short_window >= long_window:
            return -np.inf
        
        signals = self.moving_average_crossover(data, short_window, long_window)
        portfolio = self.calculate_returns(data, signals)
        returns = portfolio['returns'].dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return -np.inf
        
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        return sharpe_ratio
   
    def create_individual(self, min_window: int = 1, max_window: int = 100) -> List[int]:
        return [random.randint(min_window, max_window) for _ in range(2)]
    
    def create_population(self, min_window: int = 1, max_window: int = 100) -> List[List[int]]:
        return [self.create_individual(min_window, max_window) 
                for _ in range(self.population_size)]
    
    def tournament_selection(self, population: List[List[int]], 
                           fitnesses: List[float]) -> List[int]:
        selected_indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
        return population[best_idx].copy()
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() < self.crossover_prob:
            point = random.randint(0, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual: List[int], min_window: int = 1, 
              max_window: int = 100) -> List[int]:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_prob:
                mutated[i] = random.randint(min_window, max_window)
        return mutated
    
    def optimize(self, data: pd.DataFrame, verbose: bool = False) -> Tuple[List[int], float]:
        population = self.create_population()
        best_individual = None
        best_fitness = -np.inf
        
        for generation in range(self.generations):
            fitnesses = [self.evaluate_strategy(ind, data) for ind in population]
            
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            if verbose and generation % 10 == 0:
                valid_fitnesses = [f for f in fitnesses if f != -np.inf]
                if valid_fitnesses:
                    print(f"Generation {generation}: Best={best_fitness:.4f}, "
                          f"Avg={np.mean(valid_fitnesses):.4f}")
            
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return best_individual, best_fitness
    
    def calculate_portfolio_weights(self, assets: List[str], individual_params: Dict[str, List[int]],
                                   price_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        fitness_scores = {}
        
        for asset in assets:
            params = individual_params[asset]
            fitness = self.evaluate_strategy(params, price_data[asset])
            fitness_scores[asset] = max(0, fitness)
        
        total_fitness = sum(fitness_scores.values())
        if total_fitness == 0:
            weights = {asset: 1.0 / len(assets) for asset in assets}
        else:
            weights = {asset: score / total_fitness 
                      for asset, score in fitness_scores.items()}
        
        return weights

def optimize_portfolio(price_data: Dict[str, pd.DataFrame], 
                      population_size: int = 100,
                      generations: int = 40) -> Dict[str, List[int]]:
    ga = PortfolioGeneticAlgorithm(population_size=population_size, 
                                   generations=generations)
    
    optimized_params = {}
    for symbol, data in price_data.items():
        best_params, best_fitness = ga.optimize(data, verbose=True)
        optimized_params[symbol] = best_params
        print(f"Best parameters for {symbol}: {best_params}, Sharpe: {best_fitness:.4f}")
    
    return optimized_params
