import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def run_backtest(genes: List, trading_pairs: List[str], generation: int) -> float:
    try:
        fitness = calculate_fitness(genes, trading_pairs)
        return fitness
    except Exception as e:
        print(f"Backtest error: {e}")
        return float('-inf')

def calculate_fitness(genes: List, trading_pairs: List[str]) -> float:
    base_fitness = sum([float(g) if isinstance(g, (int, float)) else 0 for g in genes])
    pair_bonus = len(trading_pairs) * 0.1
    return base_fitness + pair_bonus

def evaluate_strategy(data: pd.DataFrame, strategy_params: dict) -> dict:
    results = {
        'total_return': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'win_rate': 0.0,
        'num_trades': 0
    }
    
    if data.empty:
        return results
        
    returns = data.get('returns', pd.Series([0]))
    results['total_return'] = returns.sum()
    results['sharpe_ratio'] = returns.mean() / (returns.std() + 1e-10)
    cumulative = (1 + returns).cumprod()
    results['max_drawdown'] = (cumulative / cumulative.cummax() - 1).min()
    
    winning_trades = (returns > 0).sum()
    total_trades = len(returns)
    results['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
    results['num_trades'] = total_trades
    
    return results
