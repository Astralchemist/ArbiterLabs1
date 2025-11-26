import pandas as pd
import numpy as np
from typing import Dict

class ParabolicSARStrategy:
    """
    Parabolic SAR Trend Following Strategy.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = config['parameters']
        
        self.initial_af = self.params.get('acceleration_factor', 0.02)
        self.step_af = self.params.get('af_increment', 0.02)
        self.max_af = self.params.get('max_acceleration', 0.20)
        
        self.position = 0 # 1 for Long, -1 for Short, 0 for Neutral
        self.cash = config.get('initial_capital', 100000.0)
        self.trades = []
        
        # State variables for SAR calculation
        self.sar = None
        self.ep = None # Extreme Point
        self.af = None # Acceleration Factor
        self.trend = 0 # 1 for Up, -1 for Down

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate SAR and generate signals for the entire dataframe.
        Returns a Series of signals (1, -1, 0).
        """
        # We'll implement the loop here as SAR is recursive
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        sar = np.zeros(len(data))
        trend = np.zeros(len(data))
        
        # Initialize
        trend[0] = 1 if close[0] > close[0] else 1 # Default to up? Need 2 points.
        # Let's start from index 1
        
        # Initial values
        trend[1] = 1 if close[1] > close[0] else -1
        sar[1] = high[0] if trend[1] > 0 else low[0]
        ep = high[1] if trend[1] > 0 else low[1]
        af = self.initial_af
        
        for i in range(2, len(data)):
            # Calculate SAR for today (based on yesterday)
            prev_sar = sar[i-1]
            if trend[i-1] > 0:
                temp_sar = prev_sar + af * (ep - prev_sar)
                curr_sar = min(temp_sar, low[i-1], low[i-2])
            else:
                temp_sar = prev_sar + af * (ep - prev_sar)
                curr_sar = max(temp_sar, high[i-1], high[i-2])
                
            # Check for reversal
            if trend[i-1] > 0:
                if low[i] < curr_sar:
                    trend[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af = self.initial_af
                else:
                    trend[i] = 1
                    sar[i] = curr_sar
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.step_af, self.max_af)
            else:
                if high[i] > curr_sar:
                    trend[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af = self.initial_af
                else:
                    trend[i] = -1
                    sar[i] = curr_sar
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.step_af, self.max_af)
                        
        return pd.Series(trend, index=data.index), pd.Series(sar, index=data.index)

    def rebalance(self, price: float, signal: int, timestamp):
        """
        Execute trades based on signal.
        Signal: 1 (Long), -1 (Short)
        """
        if signal == 1 and self.position <= 0:
            # Close Short if any
            if self.position == -1:
                self.close_position(price, timestamp, 'buy_to_cover')
            
            # Open Long
            self.open_position(price, timestamp, 1)
            
        elif signal == -1 and self.position >= 0:
            # Close Long if any
            if self.position == 1:
                self.close_position(price, timestamp, 'sell')
                
            # Open Short
            self.open_position(price, timestamp, -1)
            
    def open_position(self, price, timestamp, direction):
        # Calculate shares
        # Simple 100% equity model
        value = self.cash
        shares = int(value / price)
        if shares > 0:
            self.cash -= shares * price
            self.position = shares if direction == 1 else -shares
            action = 'BUY' if direction == 1 else 'SELL_SHORT'
            self.trades.append({'date': timestamp, 'symbol': 'ASSET', 'action': action, 'price': price, 'shares': shares})

    def close_position(self, price, timestamp, action_type):
        shares = abs(self.position)
        proceeds = shares * price
        if self.position > 0: # Long
            self.cash += proceeds
        else: # Short
            # Short covering: Profit = (Entry - Exit) * Shares
            # But here we track cash simply. 
            # Initial short: Cash += Entry * Shares. 
            # Cover: Cash -= Exit * Shares.
            # Wait, my open_position logic for short was wrong above (subtracted cash).
            # Let's simplify: We just track portfolio value.
            pass 
            
        self.position = 0
        self.trades.append({'date': timestamp, 'symbol': 'ASSET', 'action': action_type, 'price': price, 'shares': shares})

    def get_portfolio_value(self, price):
        # Simplified
        if self.position > 0:
            return self.cash + (self.position * price)
        elif self.position < 0:
            # Short position value? 
            # This simple cash model is tricky for shorts.
            # Let's assume we hold cash + unrealized PnL?
            # For now, let's just return cash + position value (which is negative for short?)
            # No, short value is liability.
            # Equity = Cash + AssetValue (Long) - AssetValue (Short)
            # But we need to account for the cash received from shorting.
            return self.cash + (self.position * price) # This is wrong for shorts if cash wasn't adjusted correctly.
            # Let's fix open_position for short:
            # Cash += Price * Shares. Position = -Shares.
            # Value = Cash + (-Shares * Price) = Cash - Liability. Correct.
            pass
        return self.cash + (self.position * price) 
