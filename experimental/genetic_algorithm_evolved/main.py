import argparse
import json
import random
from typing import List
import multiprocessing
from datetime import datetime, date
import gc

def load_trading_pairs(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config['exchange']['pair_whitelist']

def crossover_trading_pairs(parent1, parent2, num_pairs: int):
    all_pairs = list(set(parent1.trading_pairs + parent2.trading_pairs))
    if len(all_pairs) > num_pairs:
        return random.sample(all_pairs, num_pairs)
    else:
        return all_pairs

def create_population(settings, all_pairs, population_size, initial_individuals=None):
    from population import Population
    population = Population.create_random(
        size=population_size,
        parameters=settings.parameters,
        trading_pairs=all_pairs,
        num_pairs=None if settings.fix_pairs else settings.num_pairs
    )
    if initial_individuals:
        population.individuals.extend(initial_individuals)
    return population

def genetic_algorithm(settings, initial_individuals: List = None) -> List[tuple]:
    all_pairs = load_trading_pairs(settings.config_file)
    
    population_size = settings.population_size - len(initial_individuals or [])
    population = create_population(settings, all_pairs, population_size, initial_individuals)

    best_individuals = []

    with multiprocessing.Pool(processes=settings.pool_processes) as pool:
        for gen in range(settings.generations):
            print(f"Generation {gen+1}")
                        
            try:
                from operators import select_tournament, crossover, mutate
                fitnesses = pool.starmap(run_backtest, 
                    [(ind.genes, ind.trading_pairs, gen+1) for ind in population.individuals])
                
                for ind, fit in zip(population.individuals, fitnesses):
                    ind.fitness = fit if fit is not None else float('-inf')
            except Exception as e:
                print(f"Error during fitness calculation in generation {gen+1}: {str(e)}")
            
            valid_individuals = [ind for ind in population.individuals if ind.fitness is not None]
            print(f"Valid individuals in generation {gen+1}: {len(valid_individuals)}")
            if not valid_individuals:
                print(f"No valid individuals in generation {gen+1}. Terminating early.")
                break

            offspring = [select_tournament(valid_individuals, settings.tournament_size) for _ in range(settings.population_size)]

            for i in range(0, len(offspring), 2):
                if random.random() < settings.crossover_prob:
                    offspring[i], offspring[i+1] = crossover(offspring[i], offspring[i+1], with_pair=settings.fix_pairs)                    
                    offspring[i].after_genetic_operation(settings.parameters)
                    offspring[i+1].after_genetic_operation(settings.parameters)
            
            for ind in offspring:
                mutate(ind, settings.mutation_prob)
                ind.after_genetic_operation(settings.parameters)

            population.individuals = offspring

            best_individual = max(valid_individuals, key=lambda ind: ind.fitness)
            best_individuals.append((gen+1, best_individual))

            print(f"Best individual in generation {gen+1}: Fitness: {best_individual.fitness}")

            gc.collect()

    return best_individuals

def save_best_individual(individual, generation: int, settings):
    filename = f"{settings.best_generations_dir}/best_individual_gen{generation}.json"
    data = {
        'generation': generation,
        'fitness': individual.fitness,
        'genes': individual.genes,
        'trading_pairs': individual.trading_pairs
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved best individual from generation {generation} to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Run genetic algorithm for trading strategy optimization')
    parser.add_argument('--config', type=str, default='ga.json', help='Path to the configuration file')
    parser.add_argument('--download', action='store_true', help='Download data before running the algorithm')
    parser.add_argument('--start-date', type=str, default='20240101', help='Start date for data download (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default=date.today().strftime('%Y%m%d'), help='End date for data download (YYYYMMDD)')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint')
    args = parser.parse_args()

    try:
        best_individuals = genetic_algorithm(settings)

        for gen, ind in best_individuals:
            save_best_individual(ind, gen, settings)

        overall_best = max(best_individuals, key=lambda x: x[1].fitness)
        print(f"Overall best individual: Generation {overall_best[0]}, Fitness: {overall_best[1].fitness}")
        print(f"Best trading pairs: {overall_best[1].trading_pairs}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
