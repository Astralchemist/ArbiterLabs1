import random
import numpy as np
from typing import List, Tuple, Callable
from datetime import datetime, timedelta

class Gene:
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate_strength(self, stock_data: dict, date: datetime) -> float:
        return 0.0
    
    def mutate(self, mutation_rate: float = 0.1):
        if random.random() < mutation_rate:
            self.weight += random.gauss(0, 0.2)
            self.weight = max(-10, min(10, self.weight))

class Agent:
    def __init__(self, agent_id: int, genome: List[Gene]):
        self.agent_id = agent_id
        self.genome = genome
        self.fitness = 0.0
        self.validation_scores = []
    
    def calculate_stock_strength(self, stock_data: dict, date: datetime) -> float:
        return sum([gene.calculate_strength(stock_data, date) * gene.weight for gene in self.genome])
    
    def rank_stocks(self, stocks: List[dict], date: datetime) -> List[dict]:
        return sorted(stocks, key=lambda s: self.calculate_stock_strength(s, date), reverse=True)
    
    def evaluate_fitness(self, market_data: List[dict], start_date: datetime,
                        end_date: datetime, initial_capital: float = 10000) -> float:
        capital = initial_capital
        portfolio = {}
        
        current_date = start_date
        while current_date < end_date:
            ranked_stocks = self.rank_stocks(market_data, current_date)
            
            for stock in ranked_stocks[:5]:
                if capital > 0:
                    shares = capital * 0.2 / stock.get('price', 1)
                    portfolio[stock['symbol']] = portfolio.get(stock['symbol'], 0) + shares
                    capital -= shares * stock.get('price', 1)
            
            current_date += timedelta(days=1)
        
        final_value = capital
        for symbol, shares in portfolio.items():
            stock = next((s for s in market_data if s['symbol'] == symbol), None)
            if stock:
                final_value += shares * stock.get('price', 0)
        
        self.fitness = final_value / initial_capital
        return self.fitness

class TournamentSelection:
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, agents: List[Agent], num_selected: int) -> List[Agent]:
        selected = []
        for _ in range(num_selected):
            tournament = random.sample(agents, min(self.tournament_size, len(agents)))
            winner = max(tournament, key=lambda a: a.fitness)
            selected.append(winner)
        return selected

class RouletteSelection:
    def select(self, agents: List[Agent], num_selected: int) -> List[Agent]:
        total_fitness = sum(max(0, a.fitness) for a in agents)
        if total_fitness == 0:
            return random.sample(agents, num_selected)
        
        selected = []
        for _ in range(num_selected):
            spin = random.uniform(0, total_fitness)
            cumulative = 0
            for agent in agents:
                cumulative += max(0, agent.fitness)
                if cumulative >= spin:
                    selected.append(agent)
                    break
        return selected

class RankingSelection:
    def select(self, agents: List[Agent], num_selected: int) -> List[Agent]:
        sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
        return sorted_agents[:num_selected]

class Crossover:
    @staticmethod
    def single_point(parent1: Agent, parent2: Agent, offspring_id: int) -> Agent:
        point = random.randint(1, len(parent1.genome) - 1)
        child_genome = parent1.genome[:point] + parent2.genome[point:]
        return Agent(offspring_id, child_genome)
    
    @staticmethod
    def two_point(parent1: Agent, parent2: Agent, offspring_id: int) -> Agent:
        point1 = random.randint(0, len(parent1.genome) - 1)
        point2 = random.randint(point1, len(parent1.genome))
        
        child_genome = parent1.genome[:point1] + parent2.genome[point1:point2] + parent1.genome[point2:]
        return Agent(offspring_id, child_genome)
    
    @staticmethod
    def uniform(parent1: Agent, parent2: Agent, offspring_id: int) -> Agent:
        child_genome = []
        for g1, g2 in zip(parent1.genome, parent2.genome):
            child_genome.append(random.choice([g1, g2]))
        return Agent(offspring_id, child_genome)

class Mutation:
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
    
    def mutate(self, agent: Agent) -> Agent:
        for gene in agent.genome:
            gene.mutate(self.mutation_rate)
        return agent
    
    def mutate_population(self, population: List[Agent]) -> List[Agent]:
        return [self.mutate(agent) for agent in population]

class StockMarketGeneticAlgorithm:
    def __init__(self, population_size: int = 50, generations: int = 100,
                 selection_method: str = 'tournament', crossover_rate: float = 0.7,
                 mutation_rate: float = 0.1):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        if selection_method == 'tournament':
            self.selector = TournamentSelection()
        elif selection_method == 'roulette':
            self.selector = RouletteSelection()
        else:
            self.selector = RankingSelection()
        
        self.mutator = Mutation(mutation_rate)
        
        self.population = []
        self.best_agent = None
        self.fitness_history = []
    
    def initialize_population(self, genome_template: List[Gene]):
        self.population = []
        for i in range(self.population_size):
            genome = [type(gene)(weight=random.uniform(-5, 5)) for gene in genome_template]
            agent = Agent(i, genome)
            self.population.append(agent)
    
    def evolve(self, market_data: List[dict], start_date: datetime, end_date: datetime):
        for generation in range(self.generations):
            for agent in self.population:
                agent.evaluate_fitness(market_data, start_date, end_date)
            
            self.population.sort(key=lambda a: a.fitness, reverse=True)
            self.best_agent = self.population[0]
            self.fitness_history.append(self.best_agent.fitness)
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_agent.fitness:.4f}")
            
            parents = self.selector.select(self.population, self.population_size // 2)
            
            offspring = []
            next_id = len(self.population)
            for i in range(0, len(parents) - 1, 2):
                if random.random() < self.crossover_rate:
                    child1 = Crossover.single_point(parents[i], parents[i+1], next_id)
                    child2 = Crossover.single_point(parents[i+1], parents[i], next_id+1)
                    offspring.extend([child1, child2])
                    next_id += 2
            
            offspring = self.mutator.mutate_population(offspring)
            
            self.population = parents + offspring
            self.population = self.population[:self.population_size]
        
        return self.best_agent
    
    def get_best_strategy(self) -> Agent:
        return self.best_agent
