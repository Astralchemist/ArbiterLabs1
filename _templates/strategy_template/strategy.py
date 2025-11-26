"""
Strategy Template - Core Logic

This module contains the main strategy class with signal generation,
entry/exit logic, and position management.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class Strategy:
    """
    Base strategy template.

    Implement your trading logic by overriding the methods below.
    """

    def __init__(self, config: Dict):
        """
        Initialize strategy with configuration.

        Args:
            config: Dictionary containing strategy parameters
        """
        self.config = config
        self.params = config['parameters']
        self.position = 0
        self.portfolio_value = 100000  # Starting capital

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with signals: 1 (buy), -1 (sell), 0 (hold)
        """
        # TODO: Implement your signal generation logic
        signals = pd.Series(0, index=data.index)
        return signals

    def calculate_position_size(self, signal: int, current_price: float) -> float:
        """
        Calculate position size based on risk management rules.

        Args:
            signal: Trading signal (1, -1, 0)
            current_price: Current asset price

        Returns:
            Number of shares/contracts to trade
        """
        max_position = self.portfolio_value * self.config['risk']['max_position_size']
        shares = max_position / current_price
        return shares if signal != 0 else 0

    def execute_trade(self, signal: int, price: float, timestamp: pd.Timestamp):
        """
        Execute trade based on signal.

        Args:
            signal: Trading signal
            price: Execution price
            timestamp: Trade timestamp
        """
        if signal != 0:
            size = self.calculate_position_size(signal, price)
            self.position += signal * size
            # Log trade execution
            print(f"Trade: {timestamp} | Signal: {signal} | Price: {price} | Size: {size}")

    def check_exit_conditions(self, data: pd.DataFrame, current_idx: int) -> bool:
        """
        Check if exit conditions are met.

        Args:
            data: Market data
            current_idx: Current bar index

        Returns:
            True if should exit, False otherwise
        """
        # TODO: Implement exit logic
        return False
