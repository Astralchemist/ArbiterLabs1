import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

class TAGeneticAlgorithm:
    def __init__(self, price: np.ndarray, generations: int = 50,
                 population_size: int = 20, crossover_prob: float = 0.7,
                 mutation_prob: float = 0.1, method: str = 'single',
                 strategy: str = 'rsi'):
        self.price = price
        self.asset_ret = np.array(pd.DataFrame(price, columns=['return'])['return'].pct_change())
        self.generations = generations
        self.pop_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.method = method
        self.strategy = strategy
        self.best_params = None
        self.init = None
    
    def ta_initialize(self, indicator_set: Dict) -> Tuple[Dict, List]:
        init = dict()
        pop = []
        for k in range(self.pop_size):
            pop_val = []
            for i in indicator_set:
                for j in indicator_set[i]:
                    param_rng = np.random.randint(indicator_set[i][j][0], indicator_set[i][j][1])
                    pop_val.append(param_rng)
            pop.append(pop_val)
        
        init['indicators'] = list(indicator_set.keys())
        if self.method == 'single':
            init['parameters'] = [i + '_' + j for i in indicator_set for j in indicator_set[i]]
            init['initial_population'] = pop
        elif self.method == 'multiple':
            init['parameters'] = [i + '_' + j for i in indicator_set for j in indicator_set[i]] + ['exit_sig']
            pop = [(i + [np.random.randint(0, len(init['indicators']))]) for i in pop]
            init['initial_population'] = pop
        
        self.init = init
        return init, pop
    
    def calculate_rsi(self, window: int) -> np.ndarray:
        delta = np.diff(self.price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(self.price))
        avg_loss = np.zeros(len(self.price))
        
        avg_gain[window] = np.mean(gain[:window])
        avg_loss[window] = np.mean(loss[:window])
        
        for i in range(window + 1, len(self.price)):
            avg_gain[i] = (avg_gain[i-1] * (window - 1) + gain[i-1]) / window
            avg_loss[i] = (avg_loss[i-1] * (window - 1) + loss[i-1]) / window
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sma(self, window: int) -> np.ndarray:
        return pd.Series(self.price).rolling(window=window).mean().values
    
    def crossover(self, parentA: List, parentB: List) -> List:
        gene_len_a = len(parentA)
        child = np.array([np.nan for i in range(gene_len_a)])
        
        crossover_point = np.random.randint(1, gene_len_a)
        child[:crossover_point] = parentA[:crossover_point]
        child[crossover_point:] = parentB[crossover_point:]
        
        return list(child)
    
    def mutate(self, individual: List, indicator_set: Dict) -> List:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_prob:
                param_keys = list(indicator_set.keys())
                param_idx = i % len(param_keys)
                param_name = param_keys[param_idx]
                sub_keys = list(indicator_set[param_name].keys())
                sub_idx = i // len(param_keys)
                if sub_idx < len(sub_keys):
                    sub_key = sub_keys[sub_idx]
                    min_val, max_val = indicator_set[param_name][sub_key]
                    mutated[i] = np.random.randint(min_val, max_val)
        return mutated
    
    def fitness_evaluation(self, params: List) -> Tuple[float, float]:
        fitness_cumret = 0.0
        fitness_winrate = 0.0
        
        if self.strategy == 'rsi':
            window = params[0]
            buy_sig = params[1]
            sell_sig = params[2]
            
            rsi = self.calculate_rsi(window)
            signals = self.generate_rsi_signals(rsi, buy_sig, sell_sig)
            fitness_cumret, fitness_winrate = self.evaluate_signals(signals)
        
        elif self.strategy == 'sma':
            short_window = params[0]
            long_window = params[1]
            
            sma_short = self.calculate_sma(short_window)
            sma_long = self.calculate_sma(long_window)
            signals = self.generate_sma_signals(sma_short, sma_long)
            fitness_cumret, fitness_winrate = self.evaluate_signals(signals)
        
        return fitness_cumret, fitness_winrate
    
    def generate_rsi_signals(self, rsi: np.ndarray, buy_sig: float, sell_sig: float) -> np.ndarray:
        signals = np.zeros(len(self.price))
        position = 0
        
        for i in range(1, len(rsi)):
            if position == 0:
                if rsi[i] <= buy_sig:
                    signals[i] = 1
                    position = 1
                elif rsi[i] >= sell_sig:
                    signals[i] = -1
                    position = -1
            elif position == 1:
                if rsi[i] >= sell_sig:
                    signals[i] = 0
                    position = 0
                else:
                    signals[i] = 1
            elif position == -1:
                if rsi[i] <= buy_sig:
                    signals[i] = 0
                    position = 0
                else:
                    signals[i] = -1
        
        return signals
    
    def generate_sma_signals(self, sma_short: np.ndarray, sma_long: np.ndarray) -> np.ndarray:
        signals = np.zeros(len(self.price))
        position = 0
        
        for i in range(1, len(self.price)):
            if position == 0:
                if sma_short[i] > sma_long[i] and sma_short[i-1] <= sma_long[i-1]:
                    signals[i] = 1
                    position = 1
                elif sma_short[i] < sma_long[i] and sma_short[i-1] >= sma_long[i-1]:
                    signals[i] = -1
                    position = -1
            elif position == 1:
                if sma_short[i] < sma_long[i]:
                    signals[i] = 0
                    position = 0
                else:
                    signals[i] = 1
            elif position == -1:
                if sma_short[i] > sma_long[i]:
                    signals[i] = 0
                    position = 0
                else:
                    signals[i] = -1
        
        return signals
    
    def evaluate_signals(self, signals: np.ndarray) -> Tuple[float, float]:
        returns = signals[:-1] * self.asset_ret[1:]
        cumulative_return = np.prod(1 + returns[~np.isnan(returns)]) - 1
        
        trades = []
        in_trade = False
        entry_idx = 0
        
        for i in range(len(signals)):
            if not in_trade and signals[i] != 0:
                in_trade = True
                entry_idx = i
            elif in_trade and signals[i] == 0:
                in_trade = False
                trade_return = np.prod(1 + returns[entry_idx:i]) - 1
                trades.append(trade_return)
        
        win_rate = len([t for t in trades if t > 0]) / len(trades) if trades else 0
        
        return cumulative_return, win_rate
    
    def tournament_selection(self, population: List, fitnesses: List, tournament_size: int = 3) -> List:
        selected = []
        for _ in range(2):
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx])
        return selected
    
    def fit(self, indicator_set: Dict) -> List:
        init, pop = self.ta_initialize(indicator_set)
        
        best_fitness_global = float('-inf')
        best_params_global = None
        
        for generation in range(self.generations):
            fitnesses = []
            for individual in pop:
                fitness, winrate = self.fitness_evaluation(individual)
                fitnesses.append(fitness)
            
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > best_fitness_global:
                best_fitness_global = fitnesses[best_idx]
                best_params_global = pop[best_idx]
            
            new_pop = []
            for _ in range(self.pop_size):
                parents = self.tournament_selection(pop, fitnesses)
                if np.random.random() < self.crossover_prob:
                    child = self.crossover(parents[0], parents[1])
                else:
                    child = parents[0].copy()
                child = self.mutate(child, indicator_set)
                new_pop.append(child)
            
            pop = new_pop
        
        self.best_params = best_params_global
        return best_params_global
