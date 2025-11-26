import numpy as np
import pandas as pd
from typing import Dict, List

class GeneticStrategyBase:
    def __init__(self):
        self.version = "1.0.0"
        self.type = "genetic_algorithm_evolved"
        self.minimal_roi = {"0": 100}
        self.stoploss = -0.99
        self.trailing_stop = False
        self.timeframe = '5m'
        self.use_exit_signal = True
        self.exit_profit_only = False
        self.process_only_new_candles = True
        self.startup_candle_count = 168
        
    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['ema_20'] = dataframe['close'].ewm(span=20).mean()
        dataframe['ema_50'] = dataframe['close'].ewm(span=50).mean()
        dataframe['rsi'] = self.calculate_rsi(dataframe['close'], 14)
        return dataframe
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def populate_entry_trend(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe.loc[
            (dataframe['ema_20'] > dataframe['ema_50']) &
            (dataframe['rsi'] < 70),
            'enter_long'
        ] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe.loc[
            (dataframe['ema_20'] < dataframe['ema_50']) |
            (dataframe['rsi'] > 80),
            'exit_long'
        ] = 1
        return dataframe

def initialize_strategy():
    return {
        'type': 'genetic_algorithm_evolved',
        'version': '1.0.0',
        'timeframe': '5m',
        'stoploss': -0.99,
        'minimal_roi': {"0": 100}
    }

def execute(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    strategy = GeneticStrategyBase()
    data = strategy.populate_indicators(data)
    data = strategy.populate_entry_trend(data)
    data = strategy.populate_exit_trend(data)
    return data
