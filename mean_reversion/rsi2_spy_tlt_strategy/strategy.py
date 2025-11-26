import pandas as pd
import pandas_ta as ta
from typing import Dict, List

class RSI2SpyTltStrategy:
    """
    RSI(2) mean reversion on SPY and TLT.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = config['parameters']
        
        self.spy_symbol = config['data']['symbols']['equity']
        self.tlt_symbol = config['data']['symbols']['bond']
        
        self.spy_position = 0
        self.tlt_position = 0
        
        self.buy_spy_alert = False
        self.sell_spy_alert = False
        self.buy_tlt_alert = False
        self.sell_tlt_alert = False
        
        # Simple portfolio tracking
        self.initial_capital = 100000.0
        self.cash = self.initial_capital
        self.trades = []

    def generate_signals(self, spy_data: pd.DataFrame, tlt_data: pd.DataFrame):
        """
        Analyze data and update alerts.
        Expects DataFrames with 'Close'.
        """
        # Calculate RSI
        spy_rsi = ta.rsi(spy_data['Close'], length=self.params['rsi_period'])
        tlt_rsi = ta.rsi(tlt_data['Close'], length=self.params['rsi_period'])
        
        if spy_rsi is None or tlt_rsi is None or len(spy_rsi) < 1 or len(tlt_rsi) < 1:
            return

        current_spy_rsi = spy_rsi.iloc[-1]
        current_tlt_rsi = tlt_rsi.iloc[-1]
        
        # SPY Logic
        self.buy_spy_alert = False
        self.sell_spy_alert = False
        if current_spy_rsi < self.params['spy_oversold']:
            self.buy_spy_alert = True
        elif current_spy_rsi > self.params['spy_overbought']:
            self.sell_spy_alert = True
            
        # TLT Logic
        self.buy_tlt_alert = False
        self.sell_tlt_alert = False
        if current_tlt_rsi < self.params['tlt_oversold']:
            self.buy_tlt_alert = True
        elif current_tlt_rsi > self.params['tlt_overbought']:
            self.sell_tlt_alert = True
            
    def rebalance(self, spy_price: float, tlt_price: float, timestamp):
        """
        Execute trades based on alerts.
        """
        # SPY Execution
        if self.buy_spy_alert and self.spy_position == 0:
            # Buy 50% allocation
            target_value = self.get_portfolio_value(spy_price, tlt_price) * self.params['allocation_pct']
            shares = int(target_value / spy_price)
            cost = shares * spy_price
            if self.cash >= cost:
                self.cash -= cost
                self.spy_position += shares
                self.trades.append({'date': timestamp, 'symbol': self.spy_symbol, 'action': 'BUY', 'price': spy_price, 'shares': shares})
                self.buy_spy_alert = False # Reset
                
        elif self.sell_spy_alert and self.spy_position > 0:
            # Sell all
            proceeds = self.spy_position * spy_price
            self.cash += proceeds
            self.trades.append({'date': timestamp, 'symbol': self.spy_symbol, 'action': 'SELL', 'price': spy_price, 'shares': self.spy_position})
            self.spy_position = 0
            self.sell_spy_alert = False
            
        # TLT Execution
        if self.buy_tlt_alert and self.tlt_position == 0:
            # Buy 50% allocation
            target_value = self.get_portfolio_value(spy_price, tlt_price) * self.params['allocation_pct']
            shares = int(target_value / tlt_price)
            cost = shares * tlt_price
            if self.cash >= cost:
                self.cash -= cost
                self.tlt_position += shares
                self.trades.append({'date': timestamp, 'symbol': self.tlt_symbol, 'action': 'BUY', 'price': tlt_price, 'shares': shares})
                self.buy_tlt_alert = False
                
        elif self.sell_tlt_alert and self.tlt_position > 0:
            # Sell all
            proceeds = self.tlt_position * tlt_price
            self.cash += proceeds
            self.trades.append({'date': timestamp, 'symbol': self.tlt_symbol, 'action': 'SELL', 'price': tlt_price, 'shares': self.tlt_position})
            self.tlt_position = 0
            self.sell_tlt_alert = False

    def get_portfolio_value(self, spy_price, tlt_price):
        return self.cash + (self.spy_position * spy_price) + (self.tlt_position * tlt_price)