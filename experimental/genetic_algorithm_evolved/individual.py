from typing import List
import random
import copy

class Individual:
    def __init__(self, genes: List[float], trading_pairs: List[str], param_types: List[dict]):
        self.genes = genes
        self.trading_pairs = trading_pairs
        self.fitness = None
        self.param_types = param_types

    @classmethod
    def create_random(cls, parameters, all_pairs, num_pairs):
        genes = []
        for param in parameters:
            if param['type'] == 'Int':
                if param.get('name') == 'max_open_trades':
                    min_value = max(1, int(param['start']))
                    value = random.randint(min_value, int(param['end']))
                else:
                    value = random.randint(int(param['start']), int(param['end']))
            elif param['type'] == 'Decimal':
                value = random.uniform(param['start'] + 1e-10, param['end'] - 1e-10)
                value = round(value, param['decimal_places'])
            if param['type'] == 'Categorical':
                value = random.choice(param['options'])
            if param['type'] == 'Boolean':
                value = random.choice([True, False])
            genes.append(value)
        if num_pairs is not None:
            trading_pairs = random.sample(all_pairs, num_pairs)
        else:
            trading_pairs = all_pairs
        return cls(genes, trading_pairs, parameters)

    def constrain_genes(self, parameters):
        for i, param in enumerate(parameters):
            if param['type'] == 'Int':
                if param.get('name') == 'max_open_trades':
                    min_value = max(1, int(param['start']))
                    self.genes[i] = int(max(min_value, min(param['end'], self.genes[i])))
                else:
                    self.genes[i] = int(max(param['start'], min(param['end'], self.genes[i])))
            if param['type'] == 'Decimal':
                self.genes[i] = round(max(param['start'], min(param['end'], self.genes[i])), param['decimal_places'])

    def after_genetic_operation(self, parameters):
        self.constrain_genes(parameters)

    def copy(self):
        return copy.deepcopy(self)

    def mutate_trading_pairs(self, all_pairs, mutation_rate):
        if self.trading_pairs is None:
            return
        current_pairs = set(self.trading_pairs)
        
        for i in range(len(self.trading_pairs)):
            if random.random() < mutation_rate:
                available_pairs = [pair for pair in all_pairs if pair not in current_pairs]
                
                if available_pairs:
                    new_pair = random.choice(available_pairs)
                    current_pairs.remove(self.trading_pairs[i])
                    current_pairs.add(new_pair)
                    self.trading_pairs[i] = new_pair

        self.trading_pairs = list(current_pairs)
