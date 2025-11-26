"""Volatility trading strategy using RSI signals on inverse/long VIX ETFs."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import pandas_ta as ta


class VolatilityRSIStrategy:
    """
    RSI-based volatility trading with risk management.
    
    Trades SVXY (inverse VIX) and VXX (long VIX) using 2-hour RSI signals.
    Includes stop losses, take profits, and panic exits.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = config['parameters']
        
        self.short_vol_position = 0
        self.long_vol_position = 0
        self.hedge_position = 0
        
        self.short_vol_entry = 0
        self.short_vol_stop = 0
        self.long_vol_entry = 0
        self.long_vol_stop = 0
        self.long_vol_target = 0
        
        self.portfolio_value = config.get('initial_capital', 100000)
        self.cash = self.portfolio_value
        
    def resample_2h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to 2-hour bars."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df.resample('2H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    def calc_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        return ta.rsi(prices, length=period)
    
    def generate_signals(self, short_vol_data: pd.DataFrame, long_vol_data: pd.DataFrame) -> Dict:
        """Generate trading signals from RSI crossovers."""
        short_rsi2 = self.calc_rsi(short_vol_data['Close'], 2)
        short_rsi5 = self.calc_rsi(short_vol_data['Close'], 5)
        long_rsi2 = self.calc_rsi(long_vol_data['Close'], 2)
        long_rsi5 = self.calc_rsi(long_vol_data['Close'], 5)
        
        signals = {
            'short_vol_buy': False,
            'short_vol_sell': False,
            'long_vol_buy': False,
            'long_vol_sell': False,
            'short_rsi': short_rsi2.iloc[-1] if len(short_rsi2) > 0 else 50,
            'long_rsi': long_rsi2.iloc[-1] if len(long_rsi2) > 0 else 50,
        }
        
        if len(short_rsi2) < 2 or len(long_rsi2) < 2:
            return signals
        
        if (short_rsi2.iloc[-2] < 70 and short_rsi2.iloc[-1] >= 70 and 
            self.short_vol_position == 0):
            signals['short_vol_buy'] = True
        
        if (short_rsi2.iloc[-2] > 85 and short_rsi2.iloc[-1] <= 85 and 
            self.short_vol_position > 0):
            signals['short_vol_sell'] = True
        
        if (long_rsi5.iloc[-1] < 70 and
            long_rsi2.iloc[-2] > 85 and long_rsi2.iloc[-1] <= 85 and
            self.long_vol_position == 0):
            signals['long_vol_buy'] = True
        
        if (long_rsi2.iloc[-2] < 70 and long_rsi2.iloc[-1] >= 70 and
            self.long_vol_position > 0):
            signals['long_vol_sell'] = True
        
        return signals
    
    def check_panic_exit(self, data: pd.DataFrame) -> bool:
        """Check for rapid drawdown requiring immediate exit."""
        if self.short_vol_position == 0:
            return False
        
        recent_high = data['High'].tail(120).max()
        current_price = data['Close'].iloc[-1]
        drawdown = (recent_high / current_price) - 1
        
        return drawdown > 0.10
    
    def set_short_vol_stops(self, entry_price: float):
        """Set stop loss for short volatility position."""
        self.short_vol_entry = entry_price
        self.short_vol_stop = entry_price * (1 - self.params['short_vol_stop_pct'])
    
    def set_long_vol_stops(self, entry_price: float):
        """Set stop loss and take profit for long volatility position."""
        self.long_vol_entry = entry_price
        self.long_vol_stop = entry_price * (1 - self.params['long_vol_stop_pct'])
        self.long_vol_target = entry_price * (1 + self.params['long_vol_target_pct'])
    
    def check_short_vol_stop(self, current_price: float) -> bool:
        """Check if short volatility stop loss triggered."""
        if self.short_vol_position == 0:
            return False
        
        if current_price - self.short_vol_entry >= 1:
            if self.short_vol_entry > self.short_vol_stop:
                self.short_vol_stop = self.short_vol_entry
        
        return current_price <= self.short_vol_stop
    
    def check_long_vol_stops(self, current_price: float) -> Tuple[bool, str]:
        """Check if long volatility stop/target triggered."""
        if self.long_vol_position == 0:
            return False, ""
        
        if current_price <= self.long_vol_stop:
            return True, "stop_loss"
        
        if current_price >= self.long_vol_target:
            return True, "take_profit"
        
        return False, ""
    
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
            if symbol in ['XIV', 'SVXY']:
                self.short_vol_position += shares
                self.set_short_vol_stops(price)
            elif symbol == 'VXX':
                self.long_vol_position += shares
                self.set_long_vol_stops(price)
            else:
                self.hedge_position += shares
        else:
            self.cash += trade_value
            if symbol in ['XIV', 'SVXY']:
                self.short_vol_position = 0
            elif symbol == 'VXX':
                self.long_vol_position = 0
            else:
                self.hedge_position = 0
        
        return {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'value': trade_value
        }
    
    def get_portfolio_value(self, short_vol_price: float, long_vol_price: float, 
                           hedge_price: float) -> float:
        """Calculate total portfolio value."""
        return (self.cash + 
                self.short_vol_position * short_vol_price +
                self.long_vol_position * long_vol_price +
                self.hedge_position * hedge_price)
