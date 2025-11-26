"""Simple moving average crossover strategy for educational purposes."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MACrossoverStrategy:
    """
    Basic MA crossover strategy.
    
    Buy when fast MA crosses above slow MA.
    Sell when fast MA crosses below slow MA.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = config['parameters']
        
        self.position = 0
        self.invested = False
        
        self.portfolio_value = config.get('initial_capital', 100000)
        self.cash = self.portfolio_value
        self.trades = []
        
    def calc_moving_averages(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate short and long moving averages."""
        short_ma = data['Close'].rolling(window=self.params['short_window']).mean()
        long_ma = data['Close'].rolling(window=self.params['long_window']).mean()
        return short_ma, long_ma
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from MA crossovers."""
        short_ma, long_ma = self.calc_moving_averages(data)
        signals = pd.Series(0, index=data.index)
        
        if len(short_ma) < 2 or len(long_ma) < 2:
            return signals
        
        for i in range(1, len(data)):
            if (short_ma.iloc[i-1] <= long_ma.iloc[i-1] and 
                short_ma.iloc[i] > long_ma.iloc[i]):
                signals.iloc[i] = 1
            
            elif (short_ma.iloc[i-1] >= long_ma.iloc[i-1] and 
                  short_ma.iloc[i] < long_ma.iloc[i]):
                signals.iloc[i] = -1
        
        return signals
    
    def execute_trade(self, signal: int, price: float, 
                     timestamp: pd.Timestamp) -> Dict:
        """Execute trade based on signal."""
        trade = None
        
        if signal == 1 and not self.invested:
            shares = int(self.cash / price)
            
            if shares > 0:
                trade_value = shares * price
                self.cash -= trade_value
                self.position = shares
                self.invested = True
                
                trade = {
                    'timestamp': timestamp,
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'value': trade_value,
                    'cash': self.cash
                }
                
                self.trades.append(trade)
        
        elif signal == -1 and self.invested:
            shares = self.position
            trade_value = shares * price
            self.cash += trade_value
            self.position = 0
            self.invested = False
            
            trade = {
                'timestamp': timestamp,
                'action': 'sell',
                'shares': shares,
                'price': price,
                'value': trade_value,
                'cash': self.cash
            }
            
            self.trades.append(trade)
        
        return trade
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        return self.cash + (self.position * current_price)
    
    def get_positions(self) -> Dict:
        """Get current position information."""
        return {
            'shares': self.position,
            'invested': self.invested,
            'cash': self.cash
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
    
    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get indicator values for analysis."""
        short_ma, long_ma = self.calc_moving_averages(data)
        
        return pd.DataFrame({
            'price': data['Close'],
            'short_ma': short_ma,
            'long_ma': long_ma
        }, index=data.index)
