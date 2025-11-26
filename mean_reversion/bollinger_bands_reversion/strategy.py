import pandas as pd
import numpy as np

class BollingerBandsStrategy:
    def __init__(self, period=20, devfactor=2.0):
        self.period = period
        self.devfactor = devfactor
    
    def calculate_bollinger_bands(self, prices):
        sma = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        
        upper_band = sma + (std * self.devfactor)
        lower_band = sma - (std * self.devfactor)
        
        return lower_band, sma, upper_band
    
    def generate_signals(self, df):
        df = df.copy()
        lower, middle, upper = self.calculate_bollinger_bands(df['close'])
        
        df['bb_lower'] = lower
        df['bb_middle'] = middle
        df['bb_upper'] = upper
        
        df['signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'signal'] = 1
        df.loc[df['close'] > df['bb_upper'], 'signal'] = -1
        
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
            'trades': trades
        }
