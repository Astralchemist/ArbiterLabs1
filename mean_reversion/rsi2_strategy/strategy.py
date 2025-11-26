"""Mean reversion strategy using RSI(2) on SPY and TLT."""

import pandas as pd
import numpy as np
from typing import Dict, List
import pandas_ta as ta


class RSI2Strategy:
    """
    RSI(2) mean reversion on equity and bond ETFs.
    
    Buys oversold assets and sells when overbought.
    Classic Cesar Alvarez strategy.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = config['parameters']
        
        self.equity_symbol = config['symbols']['equity']
        self.bond_symbol = config['symbols']['bond']
        
        self.equity_position = 0
        self.bond_position = 0
        
        self.buy_equity_alert = False
        self.sell_equity_alert = False
        self.buy_bond_alert = False
        self.sell_bond_alert = False
        
        self.portfolio_value = config.get('initial_capital', 100000)
        self.cash = self.portfolio_value
        self.trades = []
        
    def calc_rsi(self, prices: pd.Series, period: int = 2) -> pd.Series:
        """Calculate RSI indicator."""
        return ta.rsi(prices, length=period)
    
    def set_alerts(self, equity_data: pd.DataFrame, bond_data: pd.DataFrame) -> Dict:
        """Set buy/sell alerts based on RSI levels."""
        equity_rsi = self.calc_rsi(equity_data['Close'], self.params['rsi_period'])
        bond_rsi = self.calc_rsi(bond_data['Close'], self.params['rsi_period'])
        
        self.buy_equity_alert = False
        self.sell_equity_alert = False
        self.buy_bond_alert = False
        self.sell_bond_alert = False
        
        if len(equity_rsi) < 1 or len(bond_rsi) < 1:
            return self._get_alerts()
        
        current_equity_rsi = equity_rsi.iloc[-1]
        current_bond_rsi = bond_rsi.iloc[-1]
        
        if current_equity_rsi < self.params['equity_oversold']:
            self.buy_equity_alert = True
        elif current_equity_rsi > self.params['equity_overbought']:
            self.sell_equity_alert = True
        
        if current_bond_rsi < self.params['bond_oversold']:
            self.buy_bond_alert = True
        elif current_bond_rsi > self.params['bond_overbought']:
            self.sell_bond_alert = True
        
        return self._get_alerts()
    
    def _get_alerts(self) -> Dict:
        """Get current alert status."""
        return {
            'buy_equity': self.buy_equity_alert,
            'sell_equity': self.sell_equity_alert,
            'buy_bond': self.buy_bond_alert,
            'sell_bond': self.sell_bond_alert
        }
    
    def rebalance(self, equity_price: float, bond_price: float, 
                  timestamp: pd.Timestamp) -> List[Dict]:
        """Execute trades based on alerts."""
        trades = []
        
        if self.buy_equity_alert and self.equity_position == 0:
            shares = self.calc_position_size(
                equity_price, 
                self.params['allocation_pct'] * self.params['leverage']
            )
            trade = self.execute_trade(self.equity_symbol, shares, equity_price, 'buy', timestamp)
            trades.append(trade)
            self.buy_equity_alert = False
        
        if self.sell_equity_alert and self.equity_position > 0:
            trade = self.execute_trade(
                self.equity_symbol, self.equity_position, equity_price, 'sell', timestamp
            )
            trades.append(trade)
            self.sell_equity_alert = False
        
        if self.buy_bond_alert and self.bond_position == 0:
            shares = self.calc_position_size(
                bond_price,
                self.params['allocation_pct'] * self.params['leverage']
            )
            trade = self.execute_trade(self.bond_symbol, shares, bond_price, 'buy', timestamp)
            trades.append(trade)
            self.buy_bond_alert = False
        
        if self.sell_bond_alert and self.bond_position > 0:
            trade = self.execute_trade(
                self.bond_symbol, self.bond_position, bond_price, 'sell', timestamp
            )
            trades.append(trade)
            self.sell_bond_alert = False
        
        return trades
    
    def calc_position_size(self, price: float, allocation_pct: float) -> int:
        """Calculate shares from allocation percentage."""
        allocation_value = self.portfolio_value * allocation_pct
        return int(allocation_value / price)
    
    def execute_trade(self, symbol: str, shares: int, price: float,
                     action: str, timestamp: pd.Timestamp) -> Dict:
        """Execute trade and update positions."""
        trade_value = shares * price
        
        if action == 'buy':
            self.cash -= trade_value
            if symbol == self.equity_symbol:
                self.equity_position += shares
            else:
                self.bond_position += shares
        else:
            self.cash += trade_value
            if symbol == self.equity_symbol:
                self.equity_position = 0
            else:
                self.bond_position = 0
        
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
    
    def get_portfolio_value(self, equity_price: float, bond_price: float) -> float:
        """Calculate total portfolio value."""
        return (self.cash +
                self.equity_position * equity_price +
                self.bond_position * bond_price)
    
    def get_leverage(self, equity_price: float, bond_price: float) -> float:
        """Calculate current leverage ratio."""
        portfolio_value = self.get_portfolio_value(equity_price, bond_price)
        if portfolio_value == 0:
            return 0
        
        position_value = (self.equity_position * equity_price +
                         self.bond_position * bond_price)
        
        return position_value / portfolio_value
    
    def get_positions(self) -> Dict:
        """Get current position information."""
        return {
            'equity_shares': self.equity_position,
            'bond_shares': self.bond_position,
            'cash': self.cash
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
