import pandas as pd
import numpy as np

class RsiThresholdStrategy:
    def __init__(self, rsi_period=14, rsi_low=30, rsi_high=70):
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
    
    def calculate_rsi(self, prices, period=None):
        if period is None:
            period = self.rsi_period
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, df):
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        df['signal'] = 0
        df.loc[df['rsi'] < self.rsi_low, 'signal'] = 1
        df.loc[df['rsi'] > self.rsi_high, 'signal'] = -1
        
        return df
    
    def backtest(self, df, initial_capital=10000):
        df = self.generate_signals(df)
        
        position = 0
        capital = initial_capital
        trades = []
        
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1 and position == 0:
                position = capital / df['close'].iloc[i]
                capital = 0
                trades.append({'type': 'buy', 'price': df['close'].iloc[i], 'date': df.index[i]})
            
            elif df['signal'].iloc[i] == -1 and position > 0:
                capital = position * df['close'].iloc[i]
                position = 0
                trades.append({'type': 'sell', 'price': df['close'].iloc[i], 'date': df.index[i]})
        
        final_value = capital + (position * df['close'].iloc[-1] if position > 0 else 0)
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'return': (final_value - initial_capital) / initial_capital * 100,
            'trades': trades,
            'num_trades': len(trades)
        }
    
    def optimize_parameters(self, df):
        best_return = float('-inf')
        best_params = None
        
        for period in range(7, 21):
            for low in range(20, 40, 5):
                for high in range(60, 85, 5):
                    if low >= high:
                        continue
                    
                    self.rsi_period = period
                    self.rsi_low = low
                    self.rsi_high = high
                    
                    result = self.backtest(df)
                    
                    if result['return'] > best_return:
                        best_return = result['return']
                        best_params = {
                            'rsi_period': period,
                            'rsi_low': low,
                            'rsi_high': high
                        }
        
        return best_params, best_return
