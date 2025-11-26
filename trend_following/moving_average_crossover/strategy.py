import pandas as pd
import numpy as np

class MovingAverageCrossover:
    def __init__(self, short_period=10, long_period=30):
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate_moving_averages(self, prices):
        short_ma = prices.rolling(window=self.short_period).mean()
        long_ma = prices.rolling(window=self.long_period).mean()
        return short_ma, long_ma
    
    def generate_signals(self, df):
        df = df.copy()
        df['short_ma'], df['long_ma'] = self.calculate_moving_averages(df['close'])
        
        df['signal'] = 0
        df['crossover'] = df['short_ma'] - df['long_ma']
        
        df.loc[df['crossover'] > 0, 'signal'] = 1
        df.loc[df['crossover'] < 0, 'signal'] = -1
        
        df['position'] = df['signal'].diff()
        
        return df
    
    def backtest(self, df, initial_capital=10000):
        df = self.generate_signals(df)
        
        position = 0
        capital = initial_capital
        trades = []
        
        for i in range(len(df)):
            if df['position'].iloc[i] == 1 and position == 0:
                position = capital / df['close'].iloc[i]
                capital = 0
                trades.append({'type': 'buy', 'price': df['close'].iloc[i], 'date': df.index[i]})
            
            elif df['position'].iloc[i] == -1 and position > 0:
                capital = position * df['close'].iloc[i]
                position = 0
                trades.append({'type': 'sell', 'price': df['close'].iloc[i], 'date': df.index[i]})
        
        final_value = capital + (position * df['close'].iloc[-1] if position > 0 else 0)
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'return': (final_value - initial_capital) / initial_capital * 100,
            'trades': trades
        }
    
    def optimize_parameters(self, df, short_range=(5, 20), long_range=(20, 50)):
        best_return = float('-inf')
        best_params = None
        
        for short in range(short_range[0], short_range[1] + 1):
            for long in range(long_range[0], long_range[1] + 1):
                if short >= long:
                    continue
                
                self.short_period = short
                self.long_period = long
                result = self.backtest(df)
                
                if result['return'] > best_return:
                    best_return = result['return']
                    best_params = {'short_period': short, 'long_period': long}
        
        return best_params, best_return
