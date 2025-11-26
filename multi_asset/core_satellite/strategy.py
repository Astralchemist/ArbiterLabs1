"""Core-satellite strategy combining static and momentum-based portfolios."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class CoreSatelliteStrategy:
    """
    Strategic (core) and tactical (satellite) asset allocation.
    
    Core: Fixed allocation rebalanced weekly
    Satellite: Momentum-based rotation using MA crossover
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = config['parameters']
        
        self.core_symbols = config['symbols']['core']
        self.core_weights = config['weights']['core']
        self.satellite_symbols = config['symbols']['satellite']
        
        self.positions = {symbol: 0 for symbol in 
                         self.core_symbols + self.satellite_symbols}
        
        self.portfolio_value = config.get('initial_capital', 100000)
        self.cash = self.portfolio_value
        self.last_rebalance = None
        self.trades = []
        
    def calc_moving_averages(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate fast and slow moving averages."""
        ma_fast = data['Close'].rolling(window=self.params['ma_fast']).mean()
        ma_slow = data['Close'].rolling(window=self.params['ma_slow']).mean()
        return ma_fast, ma_slow
    
    def get_momentum_positions(self, satellite_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Determine which satellite assets have positive momentum."""
        positive_momentum = []
        
        for symbol, data in satellite_data.items():
            ma_fast, ma_slow = self.calc_moving_averages(data)
            
            if len(ma_fast) < 1 or len(ma_slow) < 1:
                continue
            
            ratio = ma_fast.iloc[-1] / ma_slow.iloc[-1]
            if ratio >= 1.0:
                positive_momentum.append(symbol)
        
        return positive_momentum
    
    def should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """Check if weekly rebalance is due."""
        if self.last_rebalance is None:
            return True
        
        if current_date.weekday() != 0:
            return False
        
        days_since_rebalance = (current_date - self.last_rebalance).days
        return days_since_rebalance >= 7
    
    def rebalance_core(self, core_data: Dict[str, pd.DataFrame], 
                      timestamp: pd.Timestamp) -> List[Dict]:
        """Rebalance core portfolio to target weights."""
        trades = []
        
        for i, symbol in enumerate(self.core_symbols):
            if symbol not in core_data:
                continue
            
            current_price = core_data[symbol]['Close'].iloc[-1]
            target_weight = (self.params['core_weight'] * 
                           self.core_weights[i] * 
                           self.params['leverage'])
            
            target_value = self.portfolio_value * target_weight
            target_shares = int(target_value / current_price)
            shares_to_trade = target_shares - self.positions[symbol]
            
            if shares_to_trade != 0:
                action = 'buy' if shares_to_trade > 0 else 'sell'
                trade = self.execute_trade(
                    symbol, abs(shares_to_trade), current_price, action, timestamp
                )
                trades.append(trade)
        
        return trades
    
    def rebalance_satellite(self, satellite_data: Dict[str, pd.DataFrame],
                          timestamp: pd.Timestamp) -> List[Dict]:
        """Rebalance satellite portfolio based on momentum."""
        trades = []
        
        positive_momentum = self.get_momentum_positions(satellite_data)
        
        if len(positive_momentum) > 0:
            weight_per_position = (self.params['satellite_weight'] * 
                                  self.params['leverage'] / 
                                  len(positive_momentum))
        else:
            weight_per_position = 0
        
        for symbol in self.satellite_symbols:
            if symbol not in satellite_data:
                continue
            
            current_price = satellite_data[symbol]['Close'].iloc[-1]
            
            if symbol in positive_momentum:
                target_value = self.portfolio_value * weight_per_position
                target_shares = int(target_value / current_price)
            else:
                target_shares = 0
            
            shares_to_trade = target_shares - self.positions[symbol]
            
            if shares_to_trade != 0:
                action = 'buy' if shares_to_trade > 0 else 'sell'
                trade = self.execute_trade(
                    symbol, abs(shares_to_trade), current_price, action, timestamp
                )
                trades.append(trade)
        
        return trades
    
    def execute_trade(self, symbol: str, shares: int, price: float,
                     action: str, timestamp: pd.Timestamp) -> Dict:
        """Execute trade and update positions."""
        trade_value = shares * price
        
        if action == 'buy':
            self.cash -= trade_value
            self.positions[symbol] += shares
        else:
            self.cash += trade_value
            self.positions[symbol] -= shares
        
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'value': trade_value,
            'cash': self.cash
        }
        
        self.trades.append(trade)
        return trade
    
    def get_portfolio_value(self, all_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value."""
        position_value = 0
        
        for symbol, shares in self.positions.items():
            if shares > 0 and symbol in all_data:
                price = all_data[symbol]['Close'].iloc[-1]
                position_value += shares * price
        
        return self.cash + position_value
    
    def get_leverage(self, all_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current leverage ratio."""
        portfolio_value = self.get_portfolio_value(all_data)
        
        if portfolio_value == 0:
            return 0
        
        position_value = 0
        for symbol, shares in self.positions.items():
            if shares > 0 and symbol in all_data:
                price = all_data[symbol]['Close'].iloc[-1]
                position_value += shares * price
        
        return position_value / portfolio_value
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
