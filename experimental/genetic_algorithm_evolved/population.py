from typing import List, Dict

class Individual:
    pass

class Population:
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @classmethod
    def create_random(cls, size: int, parameters: Dict, trading_pairs: List[str], num_pairs: int):
        from individual import Individual as Ind
        return cls([Ind.create_random(parameters, trading_pairs, num_pairs) for _ in range(size)])

    def get_best(self) -> Individual:
        return max(self.individuals, key=lambda ind: ind.fitness)
